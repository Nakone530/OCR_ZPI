import sys
import os
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import (
    Button, Input, Checkbox,
    RadioSet, RadioButton,
    Static, Header, Footer, Log, Select
)
from textual.containers import Container, Vertical
from ocr.main import build_parser, main
from ocr.config import MODEL_PATH

def get_available_models():
    """Zwraca listę dostępnych modeli z folderu model_archive i głównego katalogu."""
    models = []
    
    # Model domyślny
    if os.path.exists(MODEL_PATH):
        models.append((f"Domyślny model ({MODEL_PATH})", MODEL_PATH))
    
    # Modele z folderu model_archive
    archive_dir = "./model_archive"
    if os.path.exists(archive_dir):
        for file in sorted(os.listdir(archive_dir)):
            if file.endswith('.pth'):
                full_path = os.path.join(archive_dir, file)
                models.append((file, full_path))
    
    return models

def state_to_argv(state):
    args = []

    if state["mode"] == "image":
        args += ["--image", state["input_path"]]
        
    if state["mode"] == "train":
        args += ["--train"]

    if state["mode"] == "word":
        args += ["--word", state["input_path"]]
        
    if state["denoise"]:
        args.append("--denoise")

    return args

class MyTUI(App):
    state = {
        "mode": None,
        "input_path": None,
        "model_path": MODEL_PATH,
        "epochs": 10,
        "batch_size": 32,
        "checkpoint_interval": 5,
        "resume": None,
        "denoise": False,
        "output": None,
        "output_format": "console",
        "quiet": False,
        "json": False,
        "json_pretty": False,
        "json_path": None,
    }
    def on_mount(self):
        pass

    def compose(self) -> ComposeResult:
        yield Header()

        yield Static("Wybierz tryb:")
        yield RadioSet(
            RadioButton("Train", id="train"),
            RadioButton("Infinite", id="infinite"),
            RadioButton("Image", id="image"),
            RadioButton("Word", id="word"),
        )

        yield Static("\nWybierz model:")
        models = get_available_models()
        if models:
            yield Select(
                [(name, path) for name, path in models],
                id="model_select"
            )
        
        yield Static("Lub podaj ścieżkę do modelu:")
        yield Input(placeholder="np. ./model.pth", id="custom_model_path")

        yield Checkbox("Denoise", id="denoise")

        yield Static("\nŚcieżka pliku:")
        yield Input(placeholder="np. obraz.png", id="input_path")

        yield Log(id="output_log")
        yield Button("Uruchom", id="run")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id != "run":
            return

        mode = self.query_one(RadioSet).pressed_button.id
        denoise = self.query_one("#denoise", Checkbox).value
        path = self.query_one("#input_path", Input).value
        
        # Pobierz ścieżkę modelu
        model_path = self._get_model_path()

        self.state = {
            "mode": mode,
            "denoise": denoise,
            "input_path": path,
            "model_path": model_path
        }

        parser = build_parser()
        args = parser.parse_args(state_to_argv(self.state))
        
        # Dodaj ścieżkę modelu do argumentów
        args.model_path = model_path
        
        log = self.query_one("#output_log", Log)

        # przekazujemy log jako callback do info()
        def info(msg):
            log.write_line(msg)     # wyświetla w TUI
            log_buffer.append(msg)  # zapisuje do bufora

        log_buffer = []
    
        main(args, info)

    def _get_model_path(self) -> str:
        """Pobiera wybraną ścieżkę modelu z Select lub Input."""
        # Sprawdź czy użytkownik podał custom path
        custom_path_input = self.query_one("#custom_model_path", Input)
        if custom_path_input.value.strip():
            path = custom_path_input.value.strip()
            if not os.path.exists(path):
                log = self.query_one("#output_log", Log)
                log.write_line(f"⚠️  Ostrzeżenie: Plik modelu nie istnieje: {path}")
            return path
        
        # Pobierz z Select
        try:
            select = self.query_one("#model_select", Select)
            selected = select.value
            if selected and selected != Select.BLANK:
                return selected
        except:
            pass
        
        # Fallback na domyślny model
        return MODEL_PATH
        
if __name__ == "__main__":
    MyTUI().run()
