import sys
import os
import threading
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import (
    Button, Input, Checkbox, Label,
    RadioSet, RadioButton,
    Static, Header, Footer, Log, Select
)
from textual.containers import Container, Vertical, Horizontal
from textual.screen import Screen
from textual.binding import Binding
from ocr.main import build_parser, main
from ocr.config import MODEL_PATH

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_available_models():
    """Zwraca listę dostępnych modeli z folderu model_archive i głównego katalogu."""
    models = []
    
    # Model domyślny
    if os.path.exists(MODEL_PATH):
        models.append((f"Domyślny ({MODEL_PATH})", MODEL_PATH))
    
    # Modele z folderu model_archive
    archive_dir = "./model_archive"
    if os.path.exists(archive_dir):
        for file in sorted(os.listdir(archive_dir)):
            if file.endswith('.pth'):
                full_path = os.path.join(archive_dir, file)
                models.append((file, full_path))
    
    return models

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MENU SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class MainMenuScreen(Screen):
    """Menu główne aplikacji"""
    BINDINGS = [
        Binding("q", "quit", "Wyjście"),
        Binding("r", "app_push_screen('recognize')", "Rozpoznawanie"),
        Binding("t", "app_push_screen('train')", "Trening"),
        Binding("s", "app_push_screen('settings')", "Ustawienia"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            "╔════════════════════════════════════════╗\n"
            "║       OCR - Rozpoznawanie Tekstu       ║\n"
            "║         System Menu Główne             ║\n"
            "╚════════════════════════════════════════╝",
            id="title"
        )
        yield Static("")
        yield Button("Rozpoznaj obraz/wyraz/tekst [r]", id="recognize_btn")
        yield Button("Trenuj model [t]", id="train_btn")
        yield Button("Ustawienia [s]", id="settings_btn")
        yield Button("Wyjście [q]", id="quit_btn")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "recognize_btn":
            self.app.push_screen("recognize")
        elif event.button.id == "train_btn":
            self.app.push_screen("train")
        elif event.button.id == "settings_btn":
            self.app.push_screen("settings")
        elif event.button.id == "quit_btn":
            self.app.exit()
    
    def action_app_push_screen(self, screen_id: str) -> None:
        """Action helper dla key bindings"""
        self.app.push_screen(screen_id)

# ═══════════════════════════════════════════════════════════════════════════════
# RECOGNITION SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class RecognizeScreen(Screen):
    """Ekran rozpoznawania obrazów"""
    BINDINGS = [
        Binding("escape", "back", "Powrót do menu"),
        Binding("enter", "run_recognition", "Uruchom"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(" ROZPOZNAWANIE TEKSTU", id="subtitle")
        
        yield Static("\n Wybierz typ rozpoznawania:")
        yield RadioSet(
            RadioButton("Pojedyncza litera (--image)", id="image"),
            RadioButton("Wyraz (--word)", id="word"),
            RadioButton("Wieloliniowy tekst (--lines)", id="lines"),
            RadioButton("Wiele obrazów (--multi)", id="multi"),
            id="recognition_mode"
        )
        
        yield Static("\n Ścieżka do pliku/plików:")
        yield Input(
            placeholder="Wpisz ścieżkę do obrazu np. obraz.png",
            id="input_path"
        )
        
        yield Static("\n Opcje dodatkowe:")
        yield Checkbox(" Włącz odszumianie (denoise)", id="denoise")
        yield Checkbox(" Wyjście JSON", id="json_output")
        yield Checkbox(" Zapisz do pliku", id="save_output")
        
        yield Static("Ścieżka do zapisu (opcjonalnie):")
        yield Input(
            placeholder="np. wyniki.json (lub wyniki.txt)",
            id="output_path"
        )
        
        yield Static("\n")
        with Horizontal():
            yield Button(" Uruchom [Enter]", id="recognize_run")
            yield Button(" Powrót [Esc]", id="back_btn")
        
        yield Static("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        yield Log(id="output_log")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "recognize_run":
            self._run_recognition()
        elif event.button.id == "back_btn":
            self.app.pop_screen()
    
    def action_run_recognition(self):
        """Action dla Enter key"""
        self._run_recognition()
    
    def action_back(self):
        """Action dla Escape key"""
        self.app.pop_screen()

    def _run_recognition(self):
        """Uruchom rozpoznawanie"""
        log = self.query_one("#output_log", Log)
        log.clear()
        
        # Pobierz wartości
        mode = self.query_one(RadioSet).pressed_button.id
        path_input = self.query_one("#input_path", Input).value.strip()
        denoise = self.query_one("#denoise", Checkbox).value
        json_output = self.query_one("#json_output", Checkbox).value
        save_output = self.query_one("#save_output", Checkbox).value
        output_path = self.query_one("#output_path", Input).value.strip()
        
        # Walidacja
        if not path_input:
            log.write_line(" Błąd: Podaj ścieżkę do pliku!")
            return
        
        # Obsługa multi mode
        if mode == "multi":
            # Podziel po spacjach i waliduj każdy plik
            files = path_input.split()
            missing = [f for f in files if not os.path.exists(f)]
            if missing:
                log.write_line(f" Błąd: Nie znalezione pliki: {', '.join(missing)}")
                return
            args = ["--multi"] + files
        else:
            # Dla pojedynczych plików
            if not os.path.exists(path_input):
                log.write_line(f" Błąd: Plik nie istnieje: {path_input}")
                return
            args = ["--" + mode, path_input]
        
        if denoise:
            args.append("--denoise")
        if json_output:
            args.append("--json")
            args.append("--json-pretty")
        if save_output and output_path:
            args += ["--output", output_path]
        
        log.write_line(f" Uruchamianie rozpoznawania [{mode}]...")
        log.write_line(f" Plik(i): {path_input}")
        if denoise:
            log.write_line(" Odszumianie: Włączone")
        if save_output:
            log.write_line(f" Zapis: {output_path or 'domyślna lokalizacja'}")
        log.write_line("━" * 40 + "\n")
        
        # Uruchom OCR w wątku aby nie blokować interfejsu
        def run_recognition_thread():
            try:
                parser = build_parser()
                parsed_args = parser.parse_args(args)
                
                # Użytkownik może mieć swój model w ustawieniach
                model_path = self.app.state.get("model_path", MODEL_PATH)
                parsed_args.model_path = model_path
                
                def info(msg):
                    # Thread-safe write to log
                    self.call_from_thread(log.write_line, msg)
                
                main(parsed_args, info)
                self.call_from_thread(log.write_line, "\n" + "━" * 40)
                self.call_from_thread(log.write_line, " Rozpoznawanie ukończone!")
            except Exception as e:
                self.call_from_thread(log.write_line, f"\n Błąd: {str(e)}")
                import traceback
                self.call_from_thread(log.write_line, traceback.format_exc())
        
        # Start thread
        thread = threading.Thread(target=run_recognition_thread, daemon=True)
        thread.start()

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class TrainScreen(Screen):
    """Ekran treningu modelu"""
    BINDINGS = [
        Binding("escape", "back", "Powrót do menu"),
        Binding("enter", "run_training", "Uruchom"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(" TRENING MODELU", id="subtitle")
        
        yield Static("\n Tryb treningu:")
        yield RadioSet(
            RadioButton("Zwykły trening (określona liczba epok)", id="train"),
            RadioButton("Nieskończony trening (Ctrl+C aby zatrzymać)", id="infinite"),
            id="train_mode"
        )
        
        yield Static("\n Parametry treningu:")
        yield Static("Liczba epok (epochs):", id="epochs_label")
        yield Input(
            value="10",
            id="epochs_input",
            type="integer"
        )
        
        yield Static("\nRozmiar batcha (batch-size):")
        yield Input(
            value="32",
            id="batch_size_input",
            type="integer"
        )
        
        yield Static("\nInterwał checkpointa (co ile epok zapisywać):")
        yield Input(
            value="5",
            id="checkpoint_interval_input",
            type="integer"
        )
        
        yield Static("\n Wznów z checkpointa (opcjonalnie):")
        yield Input(
            placeholder="Ścieżka do checkpoint.pth (lub pozostaw puste)",
            id="resume_input"
        )
        
        yield Static("\n")
        with Horizontal():
            yield Button(" Rozpocznij trening [Enter]", id="train_run")
            yield Button(" Powrót [Esc]", id="back_btn")
        
        yield Static("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        yield Log(id="output_log")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "train_run":
            self._run_training()
        elif event.button.id == "back_btn":
            self.app.pop_screen()
    
    def action_run_training(self):
        """Action dla Enter key"""
        self._run_training()
    
    def action_back(self):
        """Action dla Escape key"""
        self.app.pop_screen()

    def _run_training(self):
        """Uruchom trening w oddzielnym wątku"""
        log = self.query_one("#output_log", Log)
        log.clear()
        
        train_mode = self.query_one(RadioSet).pressed_button.id
        
        try:
            epochs = int(self.query_one("#epochs_input", Input).value)
            batch_size = int(self.query_one("#batch_size_input", Input).value)
            checkpoint_interval = int(self.query_one("#checkpoint_interval_input", Input).value)
            resume = self.query_one("#resume_input", Input).value.strip() or None
        except ValueError:
            log.write_line(" Błąd: Wprowadź prawidłowe liczby!")
            return
        
        # Walidacja wartości
        if epochs <= 0 or batch_size <= 0:
            log.write_line(" Błąd: Liczba epok i rozmiar batcha muszą być > 0")
            return
        
        # Buduj argumenty
        args = []
        if train_mode == "train":
            args.append("--train")
            args += ["--epochs", str(epochs)]
            args += ["--batch-size", str(batch_size)]
        else:  # infinite
            args.append("--infinite")
            args += ["--batch-size", str(batch_size)]
            args += ["--checkpoint-interval", str(checkpoint_interval)]
        
        if resume:
            if not os.path.exists(resume):
                log.write_line(f" Błąd: Checkpoint nie istnieje: {resume}")
                return
            args += ["--resume", resume]
        
        log.write_line(f" Uruchamianie treningu [{train_mode}]...")
        log.write_line(f" Epoki: {epochs}")
        log.write_line(f" Batch size: {batch_size}")
        if resume:
            log.write_line(f" Checkpoint: {resume}")
        log.write_line("━" * 40 + "\n")
        
        # Uruchom w wątku aby nie blokować interfejsu
        def run_training_thread():
            try:
                parser = build_parser()
                parsed_args = parser.parse_args(args)
                
                def info(msg):
                    # Thread-safe write to log
                    self.call_from_thread(log.write_line, msg)
                
                main(parsed_args, info)
                self.call_from_thread(log.write_line, "\n" + "━" * 40)
                self.call_from_thread(log.write_line, " Trening ukończony!")
            except KeyboardInterrupt:
                self.call_from_thread(log.write_line, "\n  Trening przerwany przez użytkownika")
            except Exception as e:
                self.call_from_thread(log.write_line, f"\n Błąd: {str(e)}")
                import traceback
                self.call_from_thread(log.write_line, traceback.format_exc())
        
        # Start thread
        thread = threading.Thread(target=run_training_thread, daemon=True)
        thread.start()

# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class SettingsScreen(Screen):
    """Ekran ustawień"""
    BINDINGS = [
        Binding("escape", "back", "Powrót do menu"),
        Binding("enter", "save_settings", "Zapisz"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(" USTAWIENIA", id="subtitle")
        
        yield Static("\n Wybierz model:")
        models = get_available_models()
        
        if models:
            yield Select(
                models,
                id="model_select",
                value=self.app.state.get("model_path", MODEL_PATH)
            )
        else:
            yield Static(" Nie znaleziono modeli!", id="no_models")
        
        yield Static("\nLub podaj niestandardową ścieżkę:")
        yield Input(
            placeholder="Ścieżka do modelu .pth",
            id="custom_model_path",
            value=self.app.state.get("model_path", MODEL_PATH)
        )
        
        yield Static("\n Domyślny format wyjścia:")
        yield RadioSet(
            RadioButton("Konsola", id="console"),
            RadioButton("Plik tekstowy", id="txt"),
            RadioButton("JSON", id="json"),
            id="output_format"
        )
        
        yield Checkbox(" Cichy tryb (bez wypisywania)", id="quiet_mode")
        
        yield Static("\n")
        with Horizontal():
            yield Button(" Zapisz [Enter]", id="save_settings")
            yield Button(" Powrót [Esc]", id="back_btn")
        
        yield Static("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        yield Log(id="info_log")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save_settings":
            self._save_settings()
        elif event.button.id == "back_btn":
            self.app.pop_screen()
    
    def action_save_settings(self):
        """Action dla Enter key"""
        self._save_settings()
    
    def action_back(self):
        """Action dla Escape key"""
        self.app.pop_screen()

    def _save_settings(self):
        """Zapisz ustawienia"""
        info_log = self.query_one("#info_log", Log)
        info_log.clear()
        
        # Pobierz ścieżkę modelu
        custom_path = self.query_one("#custom_model_path", Input).value.strip()
        
        if custom_path:
            if not os.path.exists(custom_path):
                info_log.write_line(f"  Ostrzeżenie: Plik nie istnieje: {custom_path}")
            model_path = custom_path
        else:
            try:
                select = self.query_one("#model_select", Select)
                model_path = select.value
            except:
                model_path = MODEL_PATH
        
        # Pobierz inne ustawienia
        output_format = self.query_one(RadioSet).pressed_button.id
        quiet_mode = self.query_one("#quiet_mode", Checkbox).value
        
        # Zapisz w app.state
        self.app.state["model_path"] = model_path
        self.app.state["output_format"] = output_format
        self.app.state["quiet"] = quiet_mode
        
        info_log.write_line(" Ustawienia zapisane!")
        info_log.write_line(f"    Model: {model_path}")
        info_log.write_line(f"    Format wyjścia: {output_format}")
        info_log.write_line(f"    Cichy tryb: {'Włączony' if quiet_mode else 'Wyłączony'}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

class OCRTUI(App):
    """Główna aplikacja TUI"""
    
    BINDINGS = [
        Binding("q", "quit", "Wyjście"),
    ]
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #title {
        dock: top;
        width: 100%;
        text-align: center;
        color: $accent;
        margin: 1;
    }
    
    #subtitle {
        width: 100%;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
    }
    
    Button {
        width: 100%;
        margin: 0 1 1 1;
    }
    
    RadioSet {
        margin: 1;
    }
    
    RadioButton {
        margin-left: 2;
    }
    
    Input {
        width: 100%;
        margin: 0 1 1 1;
    }
    
    Checkbox {
        margin: 0 1 1 1;
    }
    
    Log {
        height: 50%;
        width: 100%;
        margin: 1;
    }
    
    Horizontal {
        width: 100%;
        height: auto;
        margin: 0 1 1 1;
    }
    
    Horizontal > Button {
        width: 1fr;
        margin: 0 0 0 1;
    }
    
    Static {
        width: 100%;
        margin: 0 1 0 1;
    }
    """
    
    state = {
        "model_path": MODEL_PATH,
        "output_format": "console",
        "quiet": False,
    }
    
    def on_mount(self):
        """Zarejestruj wszystkie ekrany na starcie"""
        self.install_screen(MainMenuScreen(), "main")
        self.install_screen(RecognizeScreen(), "recognize")
        self.install_screen(TrainScreen(), "train")
        self.install_screen(SettingsScreen(), "settings")
        self.push_screen("main")

        
if __name__ == "__main__":
    OCRTUI().run()
