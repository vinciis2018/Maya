from typing import Optional, Dict, Any, Callable
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.box import ROUNDED

class ConsoleUI:
    """Handles console-based user interface."""
    
    def __init__(self):
        """Initialize the console UI."""
        self.console = Console()
        self._current_layout = None
    
    def display_welcome(self, agent_name: str, personalities: Dict[str, Any], 
                       voice_enabled: bool, memory_enabled: bool) -> None:
        """Display the welcome message and help.
        
        Args:
            agent_name: Name of the AI agent
            personalities: Available personalities
            voice_enabled: Whether voice output is enabled
            memory_enabled: Whether conversation memory is enabled
        """
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        
        # Header
        header_text = Text(f"✨ {agent_name} ✨", style="bold blue", justify="center")
        layout["header"].update(Panel(header_text, box=ROUNDED))
        
        # Main content
        personality_text = "\n".join(
            f"  [bold magenta]{name}[/bold magenta]: {details['description']}"
            for name, details in personalities.items()
        )
        
        memory_status = "[green]ON[/green]" if memory_enabled else "[red]OFF[/red]"
        voice_status = "[green]ON[/green]" if voice_enabled else "[red]OFF[/red]"
        
        welcome_msg = Text(f"""
Welcome to {agent_name}, your personalized AI assistant!

Memory Feature: {memory_status}
Voice Output: {voice_status}

Available Personalities:
{personality_text}

Commands:
  new - Start a new conversation
  load <id> - Load previous conversation
  switch - Change personality
  voice - Toggle voice output
  voicecmd - Use voice command instead of typing
  voicetrain - Train voice recognition
  exit - End the session
""")
        
        layout["main"].update(Panel(welcome_msg))
        self._current_layout = layout
        self.console.print(layout)
    
    def get_user_input(self, prompt: str = "> ") -> str:
        """Get input from the user.
        
        Args:
            prompt: Prompt to display
            
        Returns:
            User input as string
        """
        return input(prompt).strip()
    
    def print_message(self, message: str, style: str = None, **kwargs) -> None:
        """Print a message to the console.
        
        Args:
            message: Message to print
            style: Optional rich style to apply
            **kwargs: Additional arguments to pass to console.print()
        """
        if style:
            self.console.print(message, style=style, **kwargs)
        else:
            self.console.print(message, **kwargs)
    
    def print_error(self, message: str) -> None:
        """Print an error message.
        
        Args:
            message: Error message to print
        """
        self.console.print(f"[red]Error: {message}[/red]")
    
    def print_warning(self, message: str) -> None:
        """Print a warning message.
        
        Args:
            message: Warning message to print
        """
        self.console.print(f"[yellow]Warning: {message}[/yellow]")
        
    def show_typing(self) -> None:
        """Show a typing indicator."""
        self.console.print("Maya is typing... ", end="", style="dim")
        
    def hide_typing(self) -> None:
        """Hide the typing indicator by moving to a new line."""
        self.console.print()  # Just print a newline to clear the typing indicator
    
    def print_success(self, message: str) -> None:
        """Print a success message.
        
        Args:
            message: Success message to print
        """
        self.console.print(f"[green]✓ {message}[/green]")
        
    def print_assistant_message(self, message: str, agent_name: str = "Maya") -> None:
        """Print a message from the assistant with proper formatting.
        
        Args:
            message: The message to display
            agent_name: Name of the assistant (default: "Maya")
        """
        # Create a panel for the assistant's message
        panel = Panel(
            message,
            title=f"{agent_name}",
            title_align="left",
            border_style="blue",
            padding=(1, 2),
            expand=False
        )
        self.console.print(panel)
    
    def clear_screen(self) -> None:
        """Clear the console screen."""
        self.console.clear()
    
    def confirm(self, message: str, default: bool = False) -> bool:
        """Ask for confirmation.
        
        Args:
            message: Confirmation prompt
            default: Default value if user just presses Enter
            
        Returns:
            True if confirmed, False otherwise
        """
        suffix = " (Y/n)" if default else " (y/N)"
        while True:
            response = self.get_user_input(f"{message}{suffix} ").lower().strip()
            if not response:
                return default
            if response in ('y', 'yes'):
                return True
            if response in ('n', 'no'):
                return False
            self.print_warning("Please answer with 'y' or 'n'")
    
    def select_from_list(self, items: list, prompt: str = "Select an option") -> int:
        """Display a numbered list and let the user select an item.
        
        Args:
            items: List of items to display
            prompt: Prompt to display above the list
            
        Returns:
            Index of the selected item, or -1 if cancelled
        """
        self.console.print(f"\n{prompt}:")
        for i, item in enumerate(items, 1):
            self.console.print(f"  {i}. {item}")
            
        while True:
            try:
                choice = self.get_user_input("\nEnter number (or 'q' to cancel): ")
                if choice.lower() == 'q':
                    return -1
                    
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(items):
                    return choice_idx
                    
                self.print_warning(f"Please enter a number between 1 and {len(items)}")
            except ValueError:
                self.print_warning("Please enter a valid number")
