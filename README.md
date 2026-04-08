Assembly Vision Checker

A Python desktop application that uses YOLO object detection to verify component presence in assembly images against an expected parts list defined in an Excel file.

------------------------------------------------------------

OVERVIEW

Assembly Vision Checker is a practical computer vision tool designed for assembly verification and quality inspection.

It allows the user to:
- Load a trained YOLO model (.pt)
- Select an image of an assembly
- Provide an Excel file with expected components and quantities

The system detects components, counts them, and compares results against expectations.

Output includes:
- Bounding boxes and confidence labels
- Inventory summary (Detected vs Expected)
- GREEN border if all items are present
- RED border if any item is missing

------------------------------------------------------------

FEATURES

- Desktop GUI (Tkinter)
- YOLO object detection (Ultralytics)
- Excel-based validation
- Visual pass/fail feedback
- Inventory panel overlay
- Image preview
- Automatic output saving

------------------------------------------------------------

EXCEL FORMAT

Required columns:

Object Name | Quantity

Names must match YOLO labels exactly.

------------------------------------------------------------

OUTPUT

Processed images are saved to:

outputs/
image_name_processed.jpg

------------------------------------------------------------

RUNNING

python app/gui_detect.py

------------------------------------------------------------

TECH STACK

- Python
- Tkinter
- OpenCV
- Pandas
- Pillow
- Ultralytics YOLO

------------------------------------------------------------

AUTHOR

Taimon Miranda
