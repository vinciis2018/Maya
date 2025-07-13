import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import defaultdict, deque
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserPreferenceManager:
    """Manages learning and adapting to user preferences over time."""
    
    def __init__(self, user_id: str, data_dir: str = "data/user_preferences"):
        """Initialize the user preference manager.
        
        Args:
            user_id: Unique identifier for the user
            data_dir: Directory to store preference data
        """
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preference data
        self.preferences = {
            'response_style': {
                'formality': 0.5,  # 0.0 (casual) to 1.0 (formal)
                'verbosity': 0.5,  # 0.0 (concise) to 1.0 (detailed)
                'humor_level': 0.3,  # 0.0 (serious) to 1.0 (playful)
            },
            'topics_of_interest': defaultdict(float),  # topic -> interest_score (0.0 to 1.0)
            'preferred_response_length': 150,  # target response length in tokens
            'communication_patterns': {
                'active_hours': defaultdict(int),  # hour -> activity count
                'common_phrases': defaultdict(int),  # phrase -> usage count
                'recent_messages': deque(maxlen=100)  # Track recent messages
            },
            'last_updated': datetime.utcnow().isoformat(),
            'interaction_count': 0
        }
        
        # Load existing preferences if available
        self._load_preferences()
    
    def _get_preference_file(self) -> Path:
        """Get the path to the user's preference file."""
        return self.data_dir / f"{self.user_id}.json"
    
    def _load_preferences(self) -> None:
        """Load preferences from disk."""
        pref_file = self._get_preference_file()
        if pref_file.exists():
            try:
                with open(pref_file, 'r', encoding='utf-8') as f:
                    loaded_prefs = json.load(f)
                    self._merge_preferences(loaded_prefs)
            except Exception as e:
                logger.error(f"Error loading preferences for {self.user_id}: {str(e)}")
    
    def _merge_preferences(self, new_prefs: Dict[str, Any]) -> None:
        """Merge new preferences with existing ones."""
        for key, value in new_prefs.items():
            if key == 'topics_of_interest':
                self.preferences['topics_of_interest'].update(value)
            elif key == 'communication_patterns':
                if 'active_hours' in value:
                    self.preferences['communication_patterns']['active_hours'].update(
                        {int(k): v for k, v in value['active_hours'].items()}
                    )
                if 'common_phrases' in value:
                    self.preferences['communication_patterns']['common_phrases'].update(
                        value['common_phrases']
                    )
            elif key in self.preferences:
                if isinstance(self.preferences[key], dict) and isinstance(value, dict):
                    self.preferences[key].update(value)
                else:
                    self.preferences[key] = value
    
    def save_preferences(self) -> None:
        """Save preferences to disk."""
        try:
            pref_file = self._get_preference_file()
            with open(pref_file, 'w', encoding='utf-8') as f:
                # Convert defaultdict to regular dict for JSON serialization
                prefs_to_save = self.preferences.copy()
                prefs_to_save['topics_of_interest'] = dict(prefs_to_save['topics_of_interest'])
                prefs_to_save['communication_patterns'] = {
                    'active_hours': dict(prefs_to_save['communication_patterns']['active_hours']),
                    'common_phrases': dict(prefs_to_save['communication_patterns']['common_phrases']),
                    'recent_messages': list(prefs_to_save['communication_patterns']['recent_messages'])
                }
                json.dump(prefs_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving preferences for {self.user_id}: {str(e)}")
    
    def update_from_interaction(self, user_input: str, assistant_response: str) -> None:
        """Update preferences based on a user-assistant interaction."""
        try:
            # Update interaction count
            self.preferences['interaction_count'] += 1
            self.preferences['last_updated'] = datetime.utcnow().isoformat()
            
            # Track active hours
            current_hour = datetime.utcnow().hour
            self.preferences['communication_patterns']['active_hours'][current_hour] += 1
            
            # Track recent messages
            self.preferences['communication_patterns']['recent_messages'].append(user_input)
            
            # Update response length preference (exponential moving average)
            response_length = len(assistant_response.split())
            alpha = 0.1  # Smoothing factor
            current_avg = self.preferences['preferred_response_length']
            self.preferences['preferred_response_length'] = (
                alpha * response_length + (1 - alpha) * current_avg
            )
            
            # Analyze and update topics of interest
            self._update_topics_of_interest(user_input, assistant_response)
            
            # Analyze and update response style preferences
            self._update_response_style_preferences(user_input, assistant_response)
            
            # Save updated preferences
            self.save_preferences()
            
        except Exception as e:
            logger.error(f"Error updating preferences: {str(e)}")
    
    def _update_topics_of_interest(self, user_input: str, assistant_response: str) -> None:
        """Update topics of interest based on the conversation."""
        # Simple keyword-based topic extraction (could be enhanced with NLP)
        topic_keywords = {
            'technology': ['code', 'programming', 'tech', 'computer', 'software', 'app'],
            'sports': ['sport', 'game', 'play', 'team', 'win', 'lose', 'score'],
            'music': ['song', 'music', 'band', 'artist', 'album', 'listen'],
            'movies': ['movie', 'film', 'watch', 'actor', 'director', 'scene'],
            'food': ['food', 'eat', 'restaurant', 'meal', 'cook', 'recipe'],
            'travel': ['travel', 'trip', 'vacation', 'visit', 'place', 'country']
        }
        
        # Decay existing interests
        decay_factor = 0.99
        for topic in self.preferences['topics_of_interest']:
            self.preferences['topics_of_interest'][topic] *= decay_factor
        
        # Update interests based on current interaction
        text = f"{user_input} {assistant_response}".lower()
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    self.preferences['topics_of_interest'][topic] = min(
                        1.0, self.preferences['topics_of_interest'].get(topic, 0) + 0.05
                    )
    
    def _update_response_style_preferences(self, user_input: str, assistant_response: str) -> None:
        """Update response style preferences based on user engagement."""
        # Analyze user engagement (simplified - could use more sophisticated metrics)
        response_length = len(assistant_response.split())
        user_response_length = len(user_input.split())
        
        # If user responds with a longer message, they might prefer more detailed responses
        if user_response_length > 10 and response_length < user_response_length * 0.5:
            self.preferences['response_style']['verbosity'] = min(
                1.0, self.preferences['response_style']['verbosity'] + 0.05
            )
        
        # Detect formality indicators
        formal_indicators = ["please", "thank you", "would you", "could you"]
        formal_count = sum(1 for word in formal_indicators if word in user_input.lower())
        
        if formal_count > 0:
            self.preferences['response_style']['formality'] = min(
                1.0, self.preferences['response_style']['formality'] + 0.05
            )
        
        # Update humor preference based on user's use of humor
        humor_indicators = ["lol", "haha", "lmao", "funny", "laugh"]
        if any(indicator in user_input.lower() for indicator in humor_indicators):
            self.preferences['response_style']['humor_level'] = min(
                1.0, self.preferences['response_style']['humor_level'] + 0.05
            )
    
    def get_preferred_response_style(self) -> Dict[str, float]:
        """Get the user's preferred response style."""
        return self.preferences['response_style']
    
    def get_top_topics(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get the top n topics of interest."""
        sorted_topics = sorted(
            self.preferences['topics_of_interest'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_topics[:n]
    
    def get_preferred_response_length(self) -> int:
        """Get the user's preferred response length in tokens."""
        return int(round(self.preferences['preferred_response_length']))
    
    def get_peak_hours(self) -> List[int]:
        """Get the hours when the user is most active."""
        active_hours = self.preferences['communication_patterns']['active_hours']
        if not active_hours:
            return list(range(9, 18))  # Default to typical work hours
            
        # Get top 3 most active hours
        sorted_hours = sorted(
            active_hours.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [hour for hour, _ in sorted_hours[:3]]
