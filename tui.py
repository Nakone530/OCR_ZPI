import sys
from textual.app import App, ComposeResult
from textual.widgets import (
    Button, Input, Checkbox,
    RadioSet, RadioButton,
    Static, Header, Footer, Log
)
from ocr.main import build_parser, main

def state_to_argv(state):
    args = []

    if state["mode"] == "image":
        args += ["--image", state["input_path"]]

    if state["denoise"]:
        args.append("--denoise")

    return args

class MyTUI(App):
    state = {
        "mode": None,
        "input_path": None,
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

        yield Checkbox("Denoise", id="denoise")

        yield Static("Ścieżka pliku:")
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

        self.state = {
            "mode": mode,
            "denoise": denoise,
            "input_path": path
        }

        parser = build_parser()
        args = parser.parse_args(state_to_argv(self.state))
        log = self.query_one("#output_log", Log)

        # przekazujemy log jako callback do info()
        def info(msg):
            log.write_line(msg)     # wyświetla w TUI
            log_buffer.append(msg)  # zapisuje do bufora

        log_buffer = []
    
        main(args, info)
        
if __name__ == "__main__":
    MyTUI().run()
