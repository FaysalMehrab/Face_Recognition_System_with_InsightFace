# Face Recognition System with InsightFace

This repository provides a **face recognition system** using [InsightFace](https://github.com/deepinsight/insightface), OpenCV, and scikit-learn.  
It includes scripts for **extracting face embeddings** from images and performing **real-time face recognition** using your webcam, with results displayed via Matplotlib (for environments without OpenCV GUI support).

---

## Features

- **Face Embedding Extraction:**  
  Extracts and saves face embeddings from images organized by person.

- **Real-Time Face Recognition:**  
  Recognizes faces from your webcam feed, supporting multiple faces per frame.

- **Matplotlib Display:**  
  Uses Matplotlib for visualization, ensuring compatibility even if OpenCV GUI is not available.

- **Logging:**  
  Informative logging for all major steps and errors.

---

## Output:
![output_tom](https://github.com/user-attachments/assets/83f03e41-175a-4ad9-852c-aa380a900896)

![output_jackie](https://github.com/user-attachments/assets/ae502e8d-d00b-4ca5-a80d-c9ef460aba72)

## Folder Structure

```
.
├── Data/                # Place your training images here (one subfolder per person)
│   ├── Person1/
│   │   ├── img1.jpg
│   │   └── img2.png
│   └── Person2/
│       └── ...
├── Embeddings/          # Embeddings will be saved here after extraction
├── extract_face_embeddings.py
├── face_recognition_matplotlib.py
└── README.md
```

---

## Requirements

- Python 3.7+
- [InsightFace](https://github.com/deepinsight/insightface)
- OpenCV (`opencv-python`)
- scikit-learn
- numpy
- matplotlib

Install dependencies with:

```bash
pip install insightface opencv-python scikit-learn numpy matplotlib
```

---

## Usage

### 1. Prepare Your Data

Organize your images in the `Data/` folder, with one subfolder per person:

```
Data/
├── Alice/
│   ├── alice1.jpg
│   └── alice2.png
└── Bob/
    ├── bob1.jpg
    └── bob2.png
```

---

### 2. Extract Face Embeddings

Run the following script to extract and save embeddings for each person:

```bash
python extract_face_embeddings.py
```

- Embeddings will be saved in the `Embeddings/` folder, organized by person.

---

### 3. Real-Time Face Recognition

Run the recognition script to start your webcam and perform real-time face recognition:

```bash
python face_recognition_matplotlib.py
```

- Detected faces will be shown in a Matplotlib window with bounding boxes, names, and similarity scores.
- **Multiple faces** are supported per frame.
- **To exit:** Close the Matplotlib window or press `Ctrl+C` in the terminal.

---

### 4. Recognize a Face in a Static Image (Optional)

You can also recognize a face in a single image by calling the `main()` function in `face_recognition_matplotlib.py`:

```python
if __name__ == "__main__":
    test_image = "path_to_image.jpg"
    main(test_image)
```

---

## Notes

- The system uses **cosine similarity** for comparing face embeddings, which is standard for deep face recognition.
- If you want to use GPU, set `ctx_id=0` and use the appropriate provider in the scripts.
- For best results, use clear, front-facing images for both training and recognition.

---

## References

- [InsightFace](https://github.com/deepinsight/insightface)
- [OpenCV](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)

---

## License

This project is for educational and research purposes. Please check the licenses of the individual libraries for commercial use.
