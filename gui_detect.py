import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import pandas as pd

os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO

DEFAULT_MODEL = "models/my_model.pt"
DEFAULT_RES = "1536x2048"
DEFAULT_THRESH = 0.50

BBOX_COLORS = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
    (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)
]

def draw_inventory_panel(frame, expected_objects, detected_counts):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    def sw(x): return int(x * (w / 1280))
    def sh(y): return int(y * (h / 720))

    pad_x, pad_y = sw(16), sh(14)
    row_h = sh(26)

    rows = []
    any_missing = False

    for name, exp in expected_objects.items():
        det = int(detected_counts.get(name, 0))
        ok = det >= exp
        if not ok:
            any_missing = True
        rows.append((name, det, exp, ok))

    panel_w = int(w * 0.4)
    panel_h = sh(40 + len(rows)*30)

    overlay = frame.copy()
    cv2.rectangle(overlay, (pad_x, pad_y), (pad_x+panel_w, pad_y+panel_h), (40,40,40), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    y = pad_y + sh(30)

    for name, det, exp, ok in rows:
        color = (0,200,0) if ok else (0,0,255)
        cv2.putText(frame, f"{name}: {det}/{exp}", (pad_x+10, y),
                    font, 0.6, color, 2, cv2.LINE_AA)
        y += row_h

    return frame, any_missing


class YoloGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Assembly Vision Checker")
        self.geometry("1100x720")
        self.minsize(900,600)

        self.model = None
        self.model_path_loaded = None
        self.preview_img = None

        self.build_ui()

    def build_ui(self):
        left = ttk.Frame(self, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Label(left, text="Model").grid(row=0, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.model_var, width=40).grid(row=1, column=0)
        ttk.Button(left, text="Browse", command=self.browse_model).grid(row=1, column=1)

        self.excel_var = tk.StringVar()
        ttk.Label(left, text="Excel").grid(row=2, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.excel_var, width=40).grid(row=3, column=0)
        ttk.Button(left, text="Browse", command=self.browse_excel).grid(row=3, column=1)

        self.image_var = tk.StringVar()
        ttk.Label(left, text="Image").grid(row=4, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.image_var, width=40).grid(row=5, column=0)
        ttk.Button(left, text="Browse", command=self.browse_image).grid(row=5, column=1)

        self.run_btn = ttk.Button(left, text="Run", command=self.run_thread)
        self.run_btn.grid(row=6, column=0, pady=10)

        self.status = tk.Text(left, height=15, width=45)
        self.status.grid(row=7, column=0, columnspan=2)

        right = ttk.Frame(self)
        right.pack(fill=tk.BOTH, expand=True)

        self.preview_label = ttk.Label(right)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.status.insert(tk.END, msg + "\n")
        self.status.see(tk.END)

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("PT","*.pt")])
        if path:
            self.model_var.set(path)

    def browse_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel","*.xlsx")])
        if path:
            self.excel_var.set(path)

    def browse_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image","*.jpg;*.png")])
        if path:
            self.image_var.set(path)
            self.show_preview(path)

    def show_preview(self, path):
        img = Image.open(path)
        img.thumbnail((600,500))
        self.preview_img = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.preview_img)

    def run_thread(self):
        threading.Thread(target=self.run_detection).start()

    def run_detection(self):
        model_path = self.model_var.get()
        excel_path = self.excel_var.get()
        img_path = self.image_var.get()

        df = pd.read_excel(excel_path)
        expected = dict(zip(df["Object Name"], df["Quantity"]))

        if self.model is None:
            self.model = YOLO(model_path)

        frame = cv2.imread(img_path)
        results = self.model(frame)

        detected = {}

        for box in results[0].boxes:
            cls = int(box.cls)
            name = self.model.names[cls]
            detected[name] = detected.get(name,0)+1

        frame, missing = draw_inventory_panel(frame, expected, detected)

        thickness = 15
        if missing:
            color = (0,0,255)
        else:
            color = (0,200,0)

        cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), color, thickness)

        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", "result.jpg")
        cv2.imwrite(out_path, frame)

        self.log("Done. Saved to outputs/")

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.thumbnail((600,500))
        self.preview_img = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.preview_img)


if __name__ == "__main__":
    app = YoloGUI()
    app.mainloop()
