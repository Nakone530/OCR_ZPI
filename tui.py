import sys
import os
import json
import threading
from datetime import datetime
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import (
    Button, Input, Checkbox, Label,
    RadioSet, RadioButton,
    Static, Header, Footer, Log, Select,
    DirectoryTree, ContentSwitcher,
)
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from textual.reactive import reactive
from ocr.main import build_parser, main
from ocr.config import MODEL_PATH
from ocr.trainer import stop_training, reset_training_flags, TRAINING_PRESETS
from ocr.config import MODEL_ARCHIVE_DIR, CHECKPOINT_PATH

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_available_models():
    """Zwraca listę dostępnych modeli (domyślny + model_archive)."""
    models = []
    if os.path.exists(MODEL_PATH):
        models.append((f"Domyślny ({MODEL_PATH})", MODEL_PATH))
    archive_dir = "./model_archive"
    if os.path.exists(archive_dir):
        for file in sorted(os.listdir(archive_dir)):
            if file.endswith('.pth'):
                models.append((file, os.path.join(archive_dir, file)))
    return models

def safe_widget_id(path: str) -> str:
    """Konwertuje ścieżkę na bezpieczny identyfikator CSS."""
    return "m" + "".join(c if c.isalnum() else "_" for c in path)

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tui_settings.json")

DEFAULT_STATE: dict = {
    "model_path": MODEL_PATH,
    "selected_models": [MODEL_PATH],
    "output_format": "console",
    "quiet": False,
    "denoise_method": "nlm-color",
    "train_output_path": MODEL_PATH,
}

def load_state_from_file() -> dict:
    """Wczytuje stan z pliku JSON; zwraca DEFAULT_STATE przy błędzie."""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        state = dict(DEFAULT_STATE)
        state.update({k: v for k, v in data.items() if k in DEFAULT_STATE})
        return state
    except Exception:
        return dict(DEFAULT_STATE)

def save_state_to_file(state: dict) -> None:
    """Zapisuje stan do pliku JSON."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def model_meta(path: str) -> str:
    """Zwraca opis modelu: nazwa + rozmiar + data modyfikacji."""
    try:
        size_mb = os.path.getsize(path) / 1024 / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
        return f"{os.path.basename(path)}  [{size_mb:.1f} MB  {mtime}]"
    except Exception:
        return os.path.basename(path)

def model_label(display: str, path: str) -> str:
    """Etykieta checkboxa: nazwa wyświetlana + metadane jeśli plik istnieje."""
    if os.path.exists(path):
        return f" {display}  ({model_meta(path)})"
    return f" {display}  [brak pliku]"

# ═══════════════════════════════════════════════════════════════════════════════
# FILE BROWSER MODAL
# ═══════════════════════════════════════════════════════════════════════════════

class FileBrowserModal(ModalScreen):
    """Modal do przeglądania i wyboru plików/folderów z nawigacją w górę."""

    BINDINGS = [
        Binding("escape", "dismiss_modal", "Anuluj"),
    ]

    def __init__(
        self,
        start_path: str = ".",
        select_dirs: bool = False,
        allowed_extensions: set | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        resolved = os.path.abspath(start_path)
        if os.path.isfile(resolved):
            resolved = os.path.dirname(resolved)
        if not os.path.isdir(resolved):
            resolved = str(Path.home())
        self._current_path = resolved
        self._select_dirs = select_dirs
        self._allowed_ext: set = (
            {e.lower() for e in allowed_extensions} if allowed_extensions else set()
        )
        self._selected: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="browser-container"):
            if self._select_dirs:
                title = " WYBIERZ FOLDER"
            elif self._allowed_ext:
                exts = "  ".join(sorted(self._allowed_ext))
                title = f" WYBIERZ PLIK  [{exts}]"
            else:
                title = " WYBIERZ PLIK  — kliknij aby wybrać"
            yield Static(title, id="browser-title")
            yield Input(self._current_path, id="browser-path")
            with Horizontal(id="browser-nav"):
                yield Button("↑ Wyżej", id="browser-up")
                yield Button("Przejdź", id="browser-go", variant="primary")
                yield Button("Anuluj [Esc]", id="browser-cancel", variant="error")
            yield DirectoryTree(Path(self._current_path), id="dir_tree")
            yield Static("Wybrano: —", id="selected-label")

    def _is_allowed(self, path: str) -> bool:
        """Sprawdza czy plik ma dozwolone rozszerzenie (lub brak filtra)."""
        if not self._allowed_ext:
            return True
        return Path(path).suffix.lower() in self._allowed_ext

    def _ext_error(self, path: str) -> None:
        """Wyświetla komunikat o niedozwolonym rozszerzeniu."""
        ext = Path(path).suffix or "(brak)"
        allowed = ", ".join(sorted(self._allowed_ext))
        try:
            self.query_one("#selected-label", Static).update(
                f" Niedozwolony typ: {ext}  (dozwolone: {allowed})"
            )
        except Exception:
            pass

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        if not self._select_dirs:
            path = str(event.path)
            if not self._is_allowed(path):
                self._ext_error(path)
                return
            self.dismiss(path)

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self._selected = str(event.path)
        self._current_path = self._selected
        try:
            self.query_one("#browser-path", Input).value = self._selected
            self.query_one("#selected-label", Static).update(f"Wybrano: {self._selected}")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "browser-cancel":
            self.dismiss(None)
        elif event.button.id == "browser-go":
            typed = self.query_one("#browser-path", Input).value.strip()
            self._navigate(typed)
        elif event.button.id == "browser-up":
            parent = str(Path(self._current_path).parent)
            self._navigate(parent)

    def _navigate(self, path_str: str) -> None:
        """Przeładowuje drzewo katalogów na wskazaną ścieżkę."""
        target = Path(path_str)
        if target.is_file() and not self._select_dirs:
            if not self._is_allowed(str(target)):
                self._ext_error(str(target))
                return
            self.dismiss(str(target))
            return
        if not target.is_dir():
            try:
                self.query_one("#selected-label", Static).update(
                    f"Brak katalogu: {path_str}"
                )
            except Exception:
                pass
            return
        self._current_path = str(target)
        try:
            self.query_one("#browser-path", Input).value = self._current_path
            self.query_one("#selected-label", Static).update(f"Katalog: {self._current_path}")
            tree = self.query_one("#dir_tree", DirectoryTree)
            tree.path = target          # Path obiekt – reaktywna zmiana korzenia
        except Exception:
            pass

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION BAR
# ═══════════════════════════════════════════════════════════════════════════════

class NavBar(Static):
    """Poziomy pasek nawigacji z przyciskami stron."""

    PAGES = [
        ("1", "menu", "Menu"),
        ("2", "recognize", "Rozpoznaj"),
        ("3", "train", "Trening"),
        ("4", "settings", "Ustawienia"),
    ]

    current_page: reactive[str] = reactive("menu")

    def compose(self) -> ComposeResult:
        with Horizontal(id="nav-inner"):
            yield Static(" OCR System ", id="nav-logo")
            for key, page_id, label in self.PAGES:
                yield Button(
                    f"[{key}] {label}",
                    id=f"nav_{page_id}",
                    classes="nav-btn",
                )

    def watch_current_page(self, new_page: str) -> None:
        for _, page_id, _ in self.PAGES:
            try:
                btn = self.query_one(f"#nav_{page_id}", Button)
                if page_id == new_page:
                    btn.add_class("active")
                else:
                    btn.remove_class("active")
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        for _, page_id, _ in self.PAGES:
            if event.button.id == f"nav_{page_id}":
                self.app.navigate_to(page_id)
                event.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MENU VIEW
# ═══════════════════════════════════════════════════════════════════════════════

class MainMenuView(Static):
    """Widok menu głównego."""

    def compose(self) -> ComposeResult:
        yield Static(
            "╔══════════════════════════════════════════╗\n"
            "║       OCR - Rozpoznawanie Tekstu         ║\n"
            "║            System Menu Główne            ║\n"
            "╚══════════════════════════════════════════╝",
            id="title",
        )
        yield Static("")
        yield Static(
            "Wybierz opcję z paska nawigacji u góry lub użyj skrótów klawiszowych:\n\n"
            "  [1]  Menu główne\n"
            "  [2]  Rozpoznawanie obrazów\n"
            "  [3]  Trening modelu\n"
            "  [4]  Ustawienia\n"
            "  [q]  Wyjście z aplikacji",
            id="menu-help",
        )
        yield Static("")
        with Horizontal():
            yield Button("Rozpoznaj obraz [2]", id="menu-recognize", variant="primary")
            yield Button("Trenuj model [3]", id="menu-train")
        with Horizontal():
            yield Button("Ustawienia [4]", id="menu-settings")
            yield Button("Wyjście [q]", id="menu-quit", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "menu-recognize": "recognize",
            "menu-train": "train",
            "menu-settings": "settings",
        }
        if event.button.id in actions:
            self.app.navigate_to(actions[event.button.id])
        elif event.button.id == "menu-quit":
            self.app.exit()

# ═══════════════════════════════════════════════════════════════════════════════
# RECOGNIZE VIEW
# ═══════════════════════════════════════════════════════════════════════════════

class RecognizeView(Static):
    """Widok rozpoznawania obrazów/tekstu."""

    BINDINGS = [
        Binding("f5", "run_recognition", "Uruchom"),
    ]

    # Tryby wymagające folderu (zamiast pliku) w polu ścieżki
    _DIR_MODES = {"folder"}

    def compose(self) -> ComposeResult:
        yield Static(" ROZPOZNAWANIE TEKSTU", id="subtitle")
        yield Static("\n Wybierz typ rozpoznawania:")
        yield RadioSet(
            RadioButton("Pojedyncza litera  --image",  id="image",  value=True),
            RadioButton("Wyraz              --word",   id="word"),
            RadioButton("Tekst wielolinij.  --lines",  id="lines"),
            RadioButton("Wiele obrazów      --multi",  id="multi"),
            RadioButton("Strona → wyrazy    --page",   id="page"),
            RadioButton("Folder z obrazami  --folder", id="folder"),
            RadioButton("CRNN (tekst CNN)   --crnn",   id="crnn"),
            id="recognition_mode",
        )
        yield Static("\n Ścieżka wejściowa (plik lub folder):")
        with Horizontal():
            yield Input(
                placeholder="Wpisz ścieżkę lub kliknij Przeglądaj...",
                id="input_path",
            )
            yield Button("Przeglądaj plik",   id="browse-file",   classes="browse-btn")
            yield Button("Przeglądaj folder", id="browse-folder", classes="browse-btn")
        yield Static("\n Opcje dodatkowe:")
        yield Checkbox(" Włącz odszumianie (denoise)", id="denoise")
        yield Checkbox(" Wyjście JSON", id="json_output")
        yield Checkbox(" Zapisz do pliku", id="save_output")
        yield Checkbox(
            " Edytuj bbox interaktywnie (otwiera okno OpenCV)",
            id="bbox_edit",
            value=False,
        )
        yield Static("Ścieżka do zapisu (opcjonalnie):")
        with Horizontal():
            yield Input(
                placeholder="np. wyniki.json (lub wyniki.txt)",
                id="output_path",
            )
            yield Button("Przeglądaj", id="browse-output", classes="browse-btn")
        yield Static("\n")
        with Horizontal():
            yield Button(" Uruchom [F5]", id="recognize_run", variant="primary")
            yield Button(" Wyczyść log", id="clear_log")
        yield Static("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        yield Log(id="output_log")

    def on_mount(self) -> None:
        self.query_one("#bbox_edit", Checkbox).display = False

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "recognition_mode" and event.pressed:
            mode = event.pressed.id
            try:
                self.query_one("#bbox_edit", Checkbox).display = (mode == "page")
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "recognize_run":
            self._run_recognition()
        elif event.button.id == "clear_log":
            self.query_one("#output_log", Log).clear()
        elif event.button.id == "browse-file":
            self._browse_for(
                "input_path",
                select_dirs=False,
                allowed_extensions={".jpg", ".jpeg", ".png", ".webp"},
            )
        elif event.button.id == "browse-folder":
            self._browse_for("input_path", select_dirs=True)
        elif event.button.id == "browse-output":
            self._browse_for("output_path", select_dirs=False)

    def action_run_recognition(self) -> None:
        self._run_recognition()

    def _browse_for(
        self,
        target_id: str,
        select_dirs: bool = False,
        allowed_extensions: set | None = None,
    ) -> None:
        """Otwiera przeglądarkę plików i ustawia wynik w polu input."""
        start = self.query_one(f"#{target_id}", Input).value.strip() or "."
        if not os.path.exists(start):
            start = "."

        def on_result(path: str | None) -> None:
            if path:
                self.query_one(f"#{target_id}", Input).value = path

        self.app.push_screen(
            FileBrowserModal(
                start_path=start,
                select_dirs=select_dirs,
                allowed_extensions=allowed_extensions,
            ),
            on_result,
        )

    def _run_recognition(self) -> None:
        log = self.query_one("#output_log", Log)
        log.clear()

        radio_set = self.query_one(RadioSet)
        pressed = radio_set.pressed_button
        mode = pressed.id if pressed else "image"

        path_input = self.query_one("#input_path", Input).value.strip()
        denoise = self.query_one("#denoise", Checkbox).value
        json_output = self.query_one("#json_output", Checkbox).value
        save_output = self.query_one("#save_output", Checkbox).value
        output_path = self.query_one("#output_path", Input).value.strip()

        if not path_input:
            log.write_line(" Błąd: Podaj ścieżkę do pliku lub folderu!")
            return

        if mode == "multi":
            # Wiele plików oddzielonych spacją
            files = path_input.split()
            missing = [f for f in files if not os.path.exists(f)]
            if missing:
                log.write_line(f" Błąd: Nie znalezione pliki: {', '.join(missing)}")
                return
            base_args = ["--multi"] + files
        elif mode == "folder":
            # Tryb folderu – oczekuje katalogu
            if not os.path.isdir(path_input):
                log.write_line(f" Błąd: Nie jest folderem: {path_input}")
                return
            base_args = ["--folder", path_input]
        else:
            # Wszystkie pozostałe tryby (image, word, lines, page, crnn) – plik
            if not os.path.exists(path_input):
                log.write_line(f" Błąd: Plik nie istnieje: {path_input}")
                return
            base_args = [f"--{mode}", path_input]
            if mode == "page":
                bbox_edit = self.query_one("#bbox_edit", Checkbox).value
                if bbox_edit:
                    # Edycja bbox w osobnym oknie OpenCV, bez pytań input()
                    base_args += ["--non-interactive"]
                else:
                    # Tryb w pełni automatyczny – bez okna i bez pytań
                    base_args += ["--no-edit", "--non-interactive"]

        if denoise:
            base_args.append("--denoise")
            denoise_method = self.app.state.get("denoise_method", "nlm-color")
            base_args += ["--denoise-method", denoise_method]
        if json_output:
            base_args += ["--json", "--json-pretty"]
        if save_output and output_path:
            base_args += ["--output", output_path]

        # Pobierz listę wybranych modeli z globalnego stanu
        selected_models = self.app.state.get("selected_models", [])
        if not selected_models:
            selected_models = [self.app.state.get("model_path", MODEL_PATH)]

        log.write_line(f" Uruchamianie rozpoznawania [{mode}]...")
        log.write_line(f" Plik(i): {path_input}")
        log.write_line(f" Modele ({len(selected_models)}): {', '.join(os.path.basename(m) for m in selected_models)}")
        log.write_line("━" * 40 + "\n")

        def run_thread():
            for model_path in selected_models:
                model_name = os.path.basename(model_path)
                self.app.call_from_thread(
                    log.write_line, f"\n>>> Model: {model_name} <<<"
                )
                try:
                    parser = build_parser()
                    parsed_args = parser.parse_args(base_args[:])
                    parsed_args.model_path = model_path

                    def info(msg, _log=log):
                        self.app.call_from_thread(_log.write_line, msg)

                    main(parsed_args, info)
                except BaseException as e:
                    import traceback
                    self.app.call_from_thread(log.write_line, f" Błąd: {type(e).__name__}: {e}")
                    self.app.call_from_thread(log.write_line, traceback.format_exc())
            self.app.call_from_thread(log.write_line, "\n" + "━" * 40)
            self.app.call_from_thread(log.write_line, " Rozpoznawanie ukończone!")

        self.app._start_thread(run_thread)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN VIEW
# ═══════════════════════════════════════════════════════════════════════════════

class TrainView(Static):
    """Widok treningu modelu — zwykły / nieskończony / multi-trening z presetami."""

    BINDINGS = [
        Binding("f5", "run_training", "Uruchom"),
        Binding("f6", "stop_training", "Stop"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._running: bool = False   # instancja, nie klasa

    def compose(self) -> ComposeResult:
        yield Static(" TRENING MODELU", id="subtitle")

        # ── Tryb ────────────────────────────────────────────────
        yield Static(" Tryb treningu:")
        yield RadioSet(
            RadioButton("Zwykły  (N epok)", id="train", value=True),
            RadioButton("Nieskończony  (Stop = przerwij)", id="infinite"),
            RadioButton("Multi-trening  (kolejne presety)", id="multi_train"),
            id="train_mode",
        )

        # ── Parametry wspólne ────────────────────────────────────
        yield Static(" Rozmiar batcha:")
        yield Input(value="32", id="batch_size_input", type="integer")

        # ── Sekcja: zwykły / multi ───────────────────────────────
        with Vertical(id="section_epochs"):
            yield Static(" Liczba epok (domyślna dla multi):")
            yield Input(value="10", id="epochs_input", type="integer")

        # ── Sekcja: nieskończony ─────────────────────────────────
        with Vertical(id="section_infinite"):
            yield Static(" Interwał checkpointa (co ile epok):")
            yield Input(value="5", id="checkpoint_interval_input", type="integer")

        # ── Sekcja: presety multi-treningu ───────────────────────
        with Vertical(id="section_presets"):
            yield Static(" Wybierz presety (wielokrotny wybór):")
            with ScrollableContainer(id="preset-list"):
                for name, cfg in TRAINING_PRESETS.items():
                    yield Checkbox(
                        f" [{name}]  {cfg.description}",
                        value=True,
                        id=f"preset_{name}",
                        name=name,
                        classes="preset-cb",
                    )

        # ── Ścieżka zapisu wytrenowanego modelu ─────────────────
        yield Static(" Zapisz model do:")
        with Horizontal():
            yield Input(
                value=MODEL_PATH,
                id="save_path_input",
            )
            yield Button("Przeglądaj", id="browse-save", classes="browse-btn")

        # ── Checkpoint wznowienia ────────────────────────────────
        yield Static(" Wznów z checkpointa (opcjonalnie):")
        with Horizontal():
            yield Input(
                placeholder="Ścieżka do .pth lub pozostaw puste",
                id="resume_input",
            )
            yield Button("Przeglądaj", id="browse-resume", classes="browse-btn")

        # ── Pasek statusu ────────────────────────────────────────
        yield Static(" Status: gotowy  |  urządzenie: wykrywanie...", id="train-status")

        # ── Przyciski ────────────────────────────────────────────
        with Horizontal():
            yield Button(" Rozpocznij [F5]", id="train_run", variant="primary")
            yield Button(" Stop [F6]", id="train_stop", variant="error", disabled=True)
            yield Button(" Wyczyść log", id="clear_log")

        yield Static("━" * 40)
        yield Log(id="output_log")

    def on_mount(self) -> None:
        self._update_sections("train")
        # Ustaw ścieżkę zapisu z app.state
        try:
            self.query_one("#save_path_input", Input).value = \
                self.app.state.get("train_output_path", MODEL_PATH)
        except Exception:
            pass
        # Wykryj GPU w tle
        def detect_device():
            try:
                import torch
                dev = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
                if torch.cuda.is_available():
                    dev = f"GPU — {torch.cuda.get_device_name(0)}"
            except Exception:
                dev = "CPU"
            self.app.call_from_thread(
                self._update_status, f"gotowy  |  urządzenie: {dev}"
            )
        threading.Thread(target=detect_device, daemon=True).start()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.pressed:
            self._update_sections(event.pressed.id)

    def _update_sections(self, mode: str) -> None:
        """Pokazuje / ukrywa sekcje parametrów zależnie od trybu."""
        try:
            self.query_one("#section_epochs").display    = mode in ("train", "multi_train")
            self.query_one("#section_infinite").display  = (mode == "infinite")
            self.query_one("#section_presets").display   = (mode == "multi_train")
        except Exception:
            pass

    def _set_running(self, running: bool) -> None:
        """Blokuje / odblokowuje przyciski podczas treningu."""
        self._running = running
        try:
            self.query_one("#train_run", Button).disabled = running
            self.query_one("#train_stop", Button).disabled = not running
        except Exception:
            pass

    def _update_status(self, msg: str) -> None:
        try:
            self.query_one("#train-status", Static).update(f" {msg}")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "train_run":
            self._run_training()
        elif event.button.id == "train_stop":
            self._stop_training()
        elif event.button.id == "clear_log":
            self.query_one("#output_log", Log).clear()
        elif event.button.id == "browse-resume":
            self._browse_checkpoint()
        elif event.button.id == "browse-save":
            self._browse_save_path()

    def action_run_training(self) -> None:
        self._run_training()

    def action_stop_training(self) -> None:
        self._stop_training()

    def _browse_checkpoint(self) -> None:
        start = self.query_one("#resume_input", Input).value.strip()
        if not start or not os.path.exists(start):
            start = MODEL_ARCHIVE_DIR if os.path.isdir(MODEL_ARCHIVE_DIR) else str(Path.home())

        def on_result(path: str | None) -> None:
            if path:
                self.query_one("#resume_input", Input).value = path

        self.app.push_screen(FileBrowserModal(start_path=start, select_dirs=False), on_result)

    def _browse_save_path(self) -> None:
        start = self.query_one("#save_path_input", Input).value.strip()
        if not start or not os.path.exists(os.path.dirname(start) or "."):
            start = os.path.dirname(MODEL_PATH) or str(Path.home())

        def on_result(path: str | None) -> None:
            if path:
                self.query_one("#save_path_input", Input).value = path

        self.app.push_screen(FileBrowserModal(start_path=start, select_dirs=False), on_result)

    def _stop_training(self) -> None:
        stop_training()
        log = self.query_one("#output_log", Log)
        log.write_line(" Wysłano sygnał zatrzymania...")

    def _get_selected_presets(self) -> list[str]:
        """Zwraca nazwy zaznaczonych presetów."""
        return [
            cb.name
            for cb in self.query(".preset-cb")
            if isinstance(cb, Checkbox) and cb.value
        ]

    def _run_training(self) -> None:
        if self._running:
            return
        log = self.query_one("#output_log", Log)
        log.clear()

        radio_set = self.query_one("#train_mode", RadioSet)
        pressed = radio_set.pressed_button
        train_mode = pressed.id if pressed else "train"

        save_path = self.query_one("#save_path_input", Input).value.strip() or MODEL_PATH
        try:
            epochs = int(self.query_one("#epochs_input", Input).value)
            batch_size = int(self.query_one("#batch_size_input", Input).value)
            checkpoint_interval = int(self.query_one("#checkpoint_interval_input", Input).value)
            resume = self.query_one("#resume_input", Input).value.strip() or None
        except ValueError:
            log.write_line(" Błąd: wprowadź prawidłowe liczby!")
            return

        if batch_size <= 0:
            log.write_line(" Błąd: batch size musi być > 0")
            return
        if train_mode == "train" and epochs <= 0:
            log.write_line(" Błąd: liczba epok musi być > 0")
            return

        if resume and not os.path.exists(resume):
            log.write_line(f" Błąd: checkpoint nie istnieje: {resume}")
            return

        # Walidacja presetów dla multi_train
        presets_for_thread: list[str] = []
        if train_mode == "multi_train":
            presets_for_thread = self._get_selected_presets()
            if not presets_for_thread:
                log.write_line(" Błąd: zaznacz co najmniej jeden preset!")
                return

        log.write_line(f" Tryb: {train_mode}  |  batch: {batch_size}")
        if train_mode != "infinite":
            log.write_line(f" Epoki: {epochs}")
        if train_mode == "multi_train":
            log.write_line(f" Presety: {', '.join(presets_for_thread)}")
        log.write_line("━" * 40)

        self._set_running(True)
        self._update_status("trening w toku...")

        reset_training_flags()

        def run_thread():
            self.app.call_from_thread(log.write_line, " [wątek uruchomiony]")
            try:
                import ocr.trainer as _tr

                def info(msg):
                    epoch = _tr.GLOBAL_EPOCH
                    acc   = _tr.GLOBAL_BEST_ACC
                    self.app.call_from_thread(
                        self._update_status,
                        f"epoka {epoch}  |  best acc: {acc:.1f}%",
                    )
                    self.app.call_from_thread(log.write_line, str(msg))

                if train_mode == "train":
                    _tr.train_model(
                        epochs=epochs,
                        batch_size=batch_size,
                        model_path=resume,   # checkpoint do wznowienia
                        info=info,
                        save_path=save_path, # ścieżka zapisu modelu
                    )
                elif train_mode == "infinite":
                    _tr.infinite_train(
                        batch_size=batch_size,
                        model_path=resume,
                        checkpoint_interval=checkpoint_interval,
                        info=info,
                    )
                else:  # multi_train
                    _tr.multi_train(
                        preset_names=presets_for_thread,
                        epochs=epochs,
                        batch_size=batch_size,
                        info=info,
                    )

                self.app.call_from_thread(log.write_line, "━" * 40)
                self.app.call_from_thread(log.write_line, " Trening ukończony!")
                self.app.call_from_thread(log.write_line, f" Model: {save_path}")
            except BaseException as e:
                import traceback
                self.app.call_from_thread(log.write_line, f" Błąd: {type(e).__name__}: {e}")
                self.app.call_from_thread(log.write_line, traceback.format_exc())
            finally:
                self.app.call_from_thread(self._set_running, False)
                self.app.call_from_thread(self._update_status, "gotowy")

        self.app._start_thread(run_thread)

# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS VIEW
# ═══════════════════════════════════════════════════════════════════════════════

class SettingsView(Static):
    """Widok ustawień — modele, format, odszumianie; persystencja w JSON."""

    BINDINGS = [
        Binding("f5", "save_settings", "Zapisz"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(" USTAWIENIA", id="subtitle")

        # ── Lista modeli ────────────────────────────────────────────────────
        yield Static(" Modele do rozpoznawania (wielokrotny wybór):")
        with ScrollableContainer(id="model-list"):
            self._mount_model_checkboxes_inline()

        # ── Dodaj niestandardowy model ──────────────────────────────────────
        yield Static(" Dodaj model niestandardowy:")
        with Horizontal():
            yield Input(placeholder="Ścieżka do pliku .pth", id="custom_model_path")
            yield Button("Przeglądaj", id="browse-model", classes="browse-btn")
            yield Button("Dodaj +", id="add-model-btn")

        # ── Format wyjścia ──────────────────────────────────────────────────
        yield Static(" Domyślny format wyjścia:")
        yield RadioSet(
            RadioButton("Konsola",        id="console", value=True),
            RadioButton("Plik tekstowy",  id="txt"),
            RadioButton("JSON",           id="json"),
            id="output_format",
        )

        # ── Metoda odszumiania ──────────────────────────────────────────────
        yield Static(" Metoda odszumiania (--denoise):")
        yield RadioSet(
            RadioButton("NLM-Color  (najlepsza jakość)", id="nlm-color", value=True),
            RadioButton("Median     (szybka)",           id="median"),
            RadioButton("Bilateral  (krawędzie)",        id="bilateral"),
            RadioButton("Gaussian   (rozmycie)",         id="gaussian"),
            id="denoise_method",
        )

        yield Checkbox(" Cichy tryb (bez wypisywania na ekran)", id="quiet_mode")

        yield Static("")
        with Horizontal():
            yield Button(" Zapisz ustawienia [F5]", id="save_btn", variant="primary")
            yield Button(" Odśwież modele",         id="refresh_models")
        yield Static("━" * 40)
        yield Log(id="info_log")

    # Pomocnicze: inicjuje checkboxy wewnątrz ScrollableContainer przed mountem
    def _mount_model_checkboxes_inline(self) -> None:
        """Wywoływane z compose — yield bezpośrednio (nie mount)."""
        # Ta metoda NIE używa yield, compose wywołuje ją przez osobną metodę
        pass  # checkboxy dodajemy w on_mount

    def on_mount(self) -> None:
        """Wczytaj stan z app.state i przebuduj widgety zależne od stanu."""
        self._rebuild_model_list()
        self._apply_state_to_widgets()

    def _apply_state_to_widgets(self) -> None:
        """Ustawia kontrolki zgodnie z aktualnym app.state."""
        state = self.app.state
        try:
            fmt = state.get("output_format", "console")
            rs_fmt = self.query_one("#output_format", RadioSet)
            for btn in rs_fmt.query(RadioButton):
                btn.value = (btn.id == fmt)
        except Exception:
            pass
        try:
            method = state.get("denoise_method", "nlm-color")
            rs_den = self.query_one("#denoise_method", RadioSet)
            for btn in rs_den.query(RadioButton):
                btn.value = (btn.id == method)
        except Exception:
            pass
        try:
            self.query_one("#quiet_mode", Checkbox).value = state.get("quiet", False)
        except Exception:
            pass

    def _rebuild_model_list(self) -> None:
        """Odbudowuje listę checkboxów modeli z dysku + zapisane custom modele."""
        container = self.query_one("#model-list", ScrollableContainer)
        container.remove_children()
        models = get_available_models()
        selected = set(self.app.state.get("selected_models", [MODEL_PATH]))

        # Dodaj modele z dysku
        for display, path in models:
            lbl = model_label(display, path)
            container.mount(Checkbox(
                lbl, value=(path in selected),
                id=safe_widget_id(path), classes="model-cb", name=path,
            ))

        # Dodaj custom modele z state których nie ma na liście
        for path in self.app.state.get("selected_models", []):
            if not any(p == path for _, p in models):
                lbl = model_label(os.path.basename(path), path)
                container.mount(Checkbox(
                    lbl, value=True,
                    id=safe_widget_id(path), classes="model-cb", name=path,
                ))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save_btn":
            self._save_settings()
        elif event.button.id == "refresh_models":
            self._refresh_models()
        elif event.button.id == "browse-model":
            self._browse_model_path()
        elif event.button.id == "add-model-btn":
            self._add_custom_model()

    def action_save_settings(self) -> None:
        self._save_settings()

    def _browse_model_path(self) -> None:
        start = self.query_one("#custom_model_path", Input).value.strip()
        if not start or not os.path.exists(start):
            start = MODEL_ARCHIVE_DIR if os.path.isdir(MODEL_ARCHIVE_DIR) else str(Path.home())

        def on_result(path: str | None) -> None:
            if path:
                self.query_one("#custom_model_path", Input).value = path

        self.app.push_screen(FileBrowserModal(start_path=start, select_dirs=False), on_result)

    def _add_custom_model(self) -> None:
        log = self.query_one("#info_log", Log)
        path = self.query_one("#custom_model_path", Input).value.strip()
        if not path:
            log.write_line(" Błąd: wpisz ścieżkę modelu!")
            return

        container = self.query_one("#model-list", ScrollableContainer)
        wid = safe_widget_id(path)
        if container.query(f"#{wid}"):
            log.write_line(" Ten model jest już na liście.")
            return

        exists = os.path.exists(path)
        lbl = model_label(os.path.basename(path), path)
        container.mount(Checkbox(lbl, value=True, id=wid, classes="model-cb", name=path))
        suffix = "" if exists else "  [plik nie istnieje — sprawdź ścieżkę]"
        log.write_line(f" Dodano: {os.path.basename(path)}{suffix}")

    def _refresh_models(self) -> None:
        log = self.query_one("#info_log", Log)
        self._rebuild_model_list()
        models = get_available_models()
        log.write_line(f" Odświeżono — znaleziono {len(models)} modeli na dysku.")

    def _get_selected_models(self) -> list[str]:
        return [
            cb.name for cb in self.query(".model-cb")
            if isinstance(cb, Checkbox) and cb.value
        ]

    def _save_settings(self) -> None:
        log = self.query_one("#info_log", Log)
        log.clear()

        selected = self._get_selected_models()
        if not selected:
            log.write_line(" Ostrzeżenie: brak zaznaczonych modeli — używam domyślnego.")
            selected = [MODEL_PATH]

        fmt_pressed = self.query_one("#output_format", RadioSet).pressed_button
        output_format = fmt_pressed.id if fmt_pressed else "console"

        den_pressed = self.query_one("#denoise_method", RadioSet).pressed_button
        denoise_method = den_pressed.id if den_pressed else "nlm-color"

        quiet = self.query_one("#quiet_mode", Checkbox).value

        # Zaktualizuj app.state
        self.app.state["selected_models"]  = selected
        self.app.state["model_path"]       = selected[0]
        self.app.state["output_format"]    = output_format
        self.app.state["denoise_method"]   = denoise_method
        self.app.state["quiet"]            = quiet

        # Zapisz na dysk
        save_state_to_file(self.app.state)

        log.write_line(" Ustawienia zapisane (tui_settings.json)!")
        log.write_line(f"   Modele ({len(selected)}):")
        for m in selected:
            exists = "✓" if os.path.exists(m) else "✗"
            log.write_line(f"     {exists} {os.path.basename(m)}")
        log.write_line(f"   Format wyjścia : {output_format}")
        log.write_line(f"   Odszumianie    : {denoise_method}")
        log.write_line(f"   Cichy tryb     : {'tak' if quiet else 'nie'}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

class OCRTUI(App):
    """Główna aplikacja TUI z nawigacją stron, multi-model i przeglądarką plików."""

    BINDINGS = [
        Binding("q", "quit", "Wyjście"),
        Binding("1", "goto_page('menu')", "Menu"),
        Binding("2", "goto_page('recognize')", "Rozpoznaj"),
        Binding("3", "goto_page('train')", "Trening"),
        Binding("4", "goto_page('settings')", "Ustawienia"),
    ]

    CSS = """
    Screen {
        background: $surface;
    }

    /* ── NavBar ───────────────────────────────────────────────── */
    NavBar {
        height: 3;
        dock: top;
        background: $panel;
        padding: 0;
    }
    #nav-inner {
        height: 3;
        width: 100%;
        align: left middle;
    }
    #nav-logo {
        width: auto;
        padding: 0 2;
        color: $accent;
        text-style: bold;
    }
    .nav-btn {
        width: auto;
        min-width: 14;
        height: 3;
        margin: 0;
        border: none;
        background: $panel;
    }
    .nav-btn.active {
        background: $accent;
        color: $background;
        text-style: bold;
    }
    .nav-btn:hover {
        background: $accent 50%;
    }

    /* ── ContentSwitcher ─────────────────────────────────────── */
    ContentSwitcher {
        width: 100%;
        height: 1fr;
    }

    /* Każdy widok strony jest przewijalny i wypełnia dostępną przestrzeń */
    MainMenuView, RecognizeView, TrainView, SettingsView {
        width: 100%;
        height: 100%;
        overflow-y: auto;
    }

    /* ── Common page elements ────────────────────────────────── */
    #title {
        width: 100%;
        text-align: center;
        color: $accent;
        margin: 1;
        text-style: bold;
    }
    #subtitle {
        width: 100%;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
        text-style: bold;
    }
    #menu-help {
        margin: 1 4;
        color: $text-muted;
    }
    Button {
        width: 100%;
        margin: 0 1 1 1;
    }
    Horizontal > Button {
        width: 1fr;
        margin: 0 0 0 1;
    }
    .browse-btn {
        width: auto;
        min-width: 12;
        margin: 0 1 1 0;
    }
    RadioSet {
        margin: 1;
    }
    RadioButton {
        margin-left: 2;
    }
    Input {
        width: 1fr;
        margin: 0 1 1 1;
    }
    Checkbox {
        margin: 0 1 1 1;
    }
    Log {
        height: 15;
        width: 100%;
        margin: 1;
    }
    Static {
        width: 100%;
        margin: 0 1 0 1;
    }

    /* ── Model list ──────────────────────────────────────────── */
    #model-list {
        height: 10;
        border: solid $border;
        margin: 0 1 1 1;
        padding: 0 1;
    }
    .model-cb {
        margin: 0 0 0 1;
    }

    /* ── Train view ──────────────────────────────────────────── */
    #section_epochs, #section_infinite, #section_presets {
        width: 100%;
        height: auto;
    }
    #preset-list {
        height: 8;
        border: solid $border;
        margin: 0 1 1 1;
        padding: 0 1;
    }
    .preset-cb {
        margin: 0 0 0 1;
    }
    #train-status {
        color: $warning;
        margin: 0 1 1 1;
        text-style: bold;
    }

    /* ── File Browser Modal ───────────────────────────────────── */
    FileBrowserModal {
        align: center middle;
    }
    #browser-container {
        width: 82;
        height: 38;
        border: solid $accent;
        background: $panel;
        padding: 1 2;
    }
    #browser-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    #browser-path {
        width: 100%;
        margin-bottom: 0;
    }
    #browser-nav {
        height: 3;
        margin-bottom: 1;
    }
    #dir_tree {
        height: 1fr;
        border: solid $border;
        margin-bottom: 1;
    }
    #selected-label {
        color: $success;
        height: 1;
    }
    """

    state: dict = load_state_from_file()

    # Rejestr aktywnych wątków — potrzebny do zatrzymania przy wyjściu
    _active_threads: list = []

    def _start_thread(self, target) -> threading.Thread:
        """Uruchamia wątek i rejestruje go."""
        t = threading.Thread(target=target, daemon=True)
        self._active_threads.append(t)
        t.start()
        return t

    def action_quit(self) -> None:
        """Nadpisanie quit — sygnalizuj trening do zatrzymania przed wyjściem."""
        try:
            from ocr.trainer import stop_training
            stop_training()
        except Exception:
            pass
        self.exit()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield NavBar(id="navbar")
        yield ContentSwitcher(
            MainMenuView(id="page_menu"),
            RecognizeView(id="page_recognize"),
            TrainView(id="page_train"),
            SettingsView(id="page_settings"),
            initial="page_menu",
        )
        yield Footer()

    def navigate_to(self, page_id: str) -> None:
        """Przełącza aktywny widok i aktualizuje pasek nawigacji."""
        try:
            self.query_one(ContentSwitcher).current = f"page_{page_id}"
            self.query_one(NavBar).current_page = page_id
        except Exception:
            pass

    def action_goto_page(self, page_id: str) -> None:
        self.navigate_to(page_id)


if __name__ == "__main__":
    try:
        OCRTUI().run()
    finally:
        # Textual na Windows zostawia terminal w trybie raw — wymuszamy reset
        if sys.platform == "win32":
            import ctypes
            # Przywróć ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
            kernel32.SetConsoleMode(handle, 0x0007)
        sys.stdout.write("\033[0m\033[?25h")  # reset kolorów + pokaż kursor
        sys.stdout.flush()
        print()  # nowa linia żeby prompt CMD się pokazał
