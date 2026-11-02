import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = Path(__file__).resolve().parents[1] / "models/Qwen3-1.7B-Base"
DEFAULT_PROMPT = "The old lighthouse keeper opened the weathered journal and discovered"


class LMPlayground:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._build_window()
        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.after(100, self._poll_events)

    def _build_window(self) -> None:
        self.root.title("Base LM Playground")
        self.root.geometry("820x680")
        self.root.minsize(640, 520)

        frame = ttk.Frame(self.root, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(
            frame,
            text="Qwen3-1.7B Base",
            font=("TkDefaultFont", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame,
            text="Enter the beginning of a document. The base model will continue it.",
        ).grid(row=1, column=0, sticky="w", pady=(2, 10))

        self.prompt = scrolledtext.ScrolledText(frame, height=7, wrap=tk.WORD)
        self.prompt.grid(row=2, column=0, sticky="nsew")
        self.prompt.insert("1.0", DEFAULT_PROMPT)

        controls = ttk.Frame(frame)
        controls.grid(row=3, column=0, sticky="ew", pady=10)
        self.max_tokens = self._add_setting(controls, "Max tokens", "128", 0)
        self.temperature = self._add_setting(controls, "Temperature", "0.8", 2)
        self.top_p = self._add_setting(controls, "Top-p", "0.9", 4)

        self.generate_button = ttk.Button(
            controls,
            text="Generate",
            command=self._start_generation,
            state=tk.DISABLED,
        )
        self.generate_button.grid(row=0, column=6, padx=(18, 6))
        ttk.Button(controls, text="Clear", command=self._clear).grid(row=0, column=7)

        ttk.Label(frame, text="Continuation").grid(row=4, column=0, sticky="w")
        self.output = scrolledtext.ScrolledText(frame, wrap=tk.WORD, state=tk.DISABLED)
        self.output.grid(row=5, column=0, sticky="nsew", pady=(4, 0))

        self.status = tk.StringVar(value="Loading model...")
        ttk.Label(frame, textvariable=self.status).grid(
            row=6,
            column=0,
            sticky="w",
            pady=(8, 0),
        )
        self.root.bind("<Command-Return>", lambda _event: self._start_generation())
        self.root.bind("<Control-Return>", lambda _event: self._start_generation())

    @staticmethod
    def _add_setting(
        parent: ttk.Frame,
        label: str,
        default: str,
        column: int,
    ) -> tk.StringVar:
        value = tk.StringVar(value=default)
        ttk.Label(parent, text=label).grid(row=0, column=column, padx=(0, 4))
        ttk.Entry(parent, textvariable=value, width=7).grid(
            row=0,
            column=column + 1,
            padx=(0, 12),
        )
        return value

    def _load_model(self) -> None:
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                dtype=torch.bfloat16,
            ).to(self.device)
            self.model.eval()
            self.events.put(("loaded", None))
        except Exception as error:
            self.events.put(("error", error))

    def _start_generation(self) -> None:
        if self.model is None or self.tokenizer is None:
            return
        prompt = self.prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showerror("Missing prompt", "Enter text for the model to continue.")
            return

        try:
            max_tokens = int(self.max_tokens.get())
            temperature = float(self.temperature.get())
            top_p = float(self.top_p.get())
            if not 1 <= max_tokens <= 1024:
                raise ValueError("Max tokens must be between 1 and 1024.")
            if not 0 < temperature <= 2:
                raise ValueError("Temperature must be between 0 and 2.")
            if not 0 < top_p <= 1:
                raise ValueError("Top-p must be between 0 and 1.")
        except ValueError as error:
            messagebox.showerror("Invalid setting", str(error))
            return

        self.generate_button.configure(state=tk.DISABLED)
        self.status.set("Generating...")
        self._set_output("")
        threading.Thread(
            target=self._generate,
            args=(prompt, max_tokens, temperature, top_p),
            daemon=True,
        ).start()

    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.device)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            continuation = self.tokenizer.decode(
                output[0, inputs.input_ids.shape[-1] :],
                skip_special_tokens=True,
            )
            self.events.put(("generated", continuation))
        except Exception as error:
            self.events.put(("error", error))

    def _poll_events(self) -> None:
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if event == "loaded":
                self.status.set(f"Ready on {self.device.upper()}")
                self.generate_button.configure(state=tk.NORMAL)
            elif event == "generated":
                self._set_output(str(payload))
                self.status.set(f"Ready on {self.device.upper()}")
                self.generate_button.configure(state=tk.NORMAL)
            elif event == "error":
                self.status.set("Error")
                state = tk.NORMAL if self.model is not None else tk.DISABLED
                self.generate_button.configure(state=state)
                messagebox.showerror("Model error", str(payload))
        self.root.after(100, self._poll_events)

    def _set_output(self, text: str) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.insert("1.0", text)
        self.output.configure(state=tk.DISABLED)

    def _clear(self) -> None:
        self.prompt.delete("1.0", tk.END)
        self._set_output("")


def main() -> None:
    root = tk.Tk()
    LMPlayground(root)
    root.mainloop()


if __name__ == "__main__":
    main()
