import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import gradio as gr

from myMetroProcessing import load_models, processOneMetroImage

# Charger les modèles une seule fois au démarrage
load_models()

def predict_metro_lines(image: Image.Image, resize_factor: float = 1.0):
    # Conversion en numpy (RGB)
    im = np.array(image.convert("RGB"))

    # Appel de la fonction projet
    im_resized, bd = processOneMetroImage("uploaded", im, 0, resize_factor)

    # bd est un tableau [n, y1, y2, x1, x2, classe]
    # On dessine les rectangles + labels sur l’image
    fig, ax = plt.subplots()
    ax.imshow(im_resized)
    ax.axis("off")

    if bd is not None and bd.shape[0] > 0:
        for row in bd:
            _, y1, y2, x1, x2, classe = row
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"Ligne {classe}",
                    fontsize=8, bbox=dict(facecolor="white", alpha=0.7))

    # Conversion de la figure matplotlib en image PIL
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    out_image = Image.open(buf)

    # Tableau de résultats pour affichage dans Gradio
    if bd is not None and bd.shape[0] > 0:
        df = pd.DataFrame(bd, columns=["image_idx", "y1", "y2", "x1", "x2", "line"])
    else:
        df = pd.DataFrame(columns=["image_idx", "y1", "y2", "x1", "x2", "line"])

    return out_image, df

demo = gr.Interface(
    fn=predict_metro_lines,
    inputs=[
        gr.Image(type="pil", label="Image de panneau de métro"),
        gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Facteur de redimensionnement")
    ],
    outputs=[
        gr.Image(label="Image annotée"),
        gr.Dataframe(label="Lignes détectées")
    ],
    title="Détection de lignes de métro parisiennes",
    description="Upload une photo de panneau de métro, le modèle détecte les pictogrammes et prédit la ligne."
)

if __name__ == "__main__":
    demo.launch()
