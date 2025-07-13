<!-- start environment -->
source venv/bin/activate

<!-- Run installation if first time -->
pip install -r requirements.txt

<!-- Run Ollama -->
ollam server

<!-- Run Flask -->
python app.py

<!-- Run CLI Client -->
python cli_client.py
<!-- IN CLI -->
voice = start voice output
voicetrain = train for voice recognition
voicecmd = use voice command instead of typing
killall say = stop all ongoing speech
<!-- Change voice name in config.json -->
Natural Female = Samantha
British = Daniel
Dramatic = Zarvox
Softer = Whisper
robotic = Organ

Bad News            en_US    # Hello! My name is Bad News.
Bahh                en_US    # Hello! My name is Bahh.
Bells               en_US    # Hello! My name is Bells.
Boing               en_US    # Hello! My name is Boing.
Bubbles             en_US    # Hello! My name is Bubbles.
Albert              en_US    # Hello! My name is Albert.
Cellos              en_US    # Hello! My name is Cellos.
Wobble              en_US    # Hello! My name is Wobble.
Eddy (English (UK)) en_GB    # Hello! My name is Eddy.
Eddy (English (US)) en_US    # Hello! My name is Eddy.
Flo (English (UK))  en_GB    # Hello! My name is Flo.
Flo (English (US))  en_US    # Hello! My name is Flo.
Fred                en_US    # Hello! My name is Fred.
Good News           en_US    # Hello! My name is Good News.
Jester              en_US    # Hello! My name is Jester.
Junior              en_US    # Hello! My name is Junior.
Karen               en_AU    # Hi my name is Karen
Kathy               en_US    # Hello! My name is Kathy.
Lekha               hi_IN    # नमस्ते, मेरा नाम लेखा है।
Moira               en_IE    # Hello! My name is Moira.
Superstar           en_US    # Hello! My name is Superstar.
Ralph               en_US    # Hello! My name is Ralph.
Reed (English (UK)) en_GB    # Hello! My name is Reed.
Reed (English (US)) en_US    # Hello! My name is Reed.
Rishi               en_IN    # Hello! My name is Rishi.
Rocko (English (UK)) en_GB    # Hello! My name is Rocko.
Rocko (English (US)) en_US    # Hello! My name is Rocko.
Sandy (English (UK)) en_GB    # Hello! My name is Sandy.
Sandy (English (US)) en_US    # Hello! My name is Sandy.
Shelley (English (UK)) en_GB    # Hello! My name is Shelley.
Shelley (English (US)) en_US    # Hello! My name is Shelley.
Tessa               en_ZA    # Hello! My name is Tessa.
Trinoids            en_US    # Hello! My name is Trinoids.
Vani                ta_IN    # Hello! My name is Vani.


<!-- Run Voice to Text Command -->
python voice_command.py


<!-- Upload a document for RAG -->
curl -X POST -F "file=@test.pdf" http://localhost:3009/api/documents/upload

<!-- Process a web page for RAG-->
curl -X POST -H "Content-Type: application/json" -d '{"url":"https://example.com"}' http://localhost:3009/api/documents/process-url


<!-- Process a directory for RAG -->
curl -X POST -H "Content-Type: application/json" -d '{"directory":"/Users/kishanvinciis/Downloads/INVOICE_WEEDINGWIRE_Weedingwire060625"}' http://localhost:3009/api/documents/process-directory