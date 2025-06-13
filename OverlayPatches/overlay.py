"""
Module for generating overlay visualizations of segmentation masks on images.

Provides functions to overlay ground-truth and predicted masks on an input image,
highlighting true positives in green and errors (false positives and false negatives)
in red, then save and display the results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def overlay_masks(
    img: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create overlay images showing true positives and errors.

    Args:
        img (np.ndarray): Original image in BGR format, shape (H, W, 3).
        gt_mask (np.ndarray): Ground-truth mask, 8-bit grayscale, shape (H, W).
        pred_mask (np.ndarray): Predicted mask, 8-bit grayscale, shape (H, W).
        alpha (float): Opacity for overlay blending (0.0 transparent to 1.0 opaque).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            img_tp:     Overlay showing only true positives in green.
            img_err:    Overlay showing only errors (FP + FN) in red.
            img_both:   Overlay combining TP (verde) e erros (vermelho).

    Raises:
        ValueError: If `img`, `gt_mask` e `pred_mask` não tiverem dimensões compatíveis.
    """
    # Verifica compatibilidade de dimensões
    if img.shape[:2] != gt_mask.shape or img.shape[:2] != pred_mask.shape:
        raise ValueError("Image and masks must have the same height and width")

    # Binariza máscaras (0 ou 1)
    _, gt = cv2.threshold(gt_mask,   127, 1, cv2.THRESH_BINARY)
    _, pred = cv2.threshold(pred_mask, 127, 1, cv2.THRESH_BINARY)

    # True positives: ambos têm valor 1
    correct = (pred == gt) & (gt == 1)
    # Erros: predição diferente da ground-truth
    wrong = (pred != gt)

    h, w = img.shape[:2]
    base = np.zeros((h, w, 3), dtype=np.uint8)

    # 1) Apenas TP em verde
    overlay_tp = base.copy()
    overlay_tp[correct] = (0, 255, 0)
    img_tp = cv2.addWeighted(img, 1 - alpha, overlay_tp, alpha, 0)

    # 2) Apenas erros em vermelho
    overlay_err = base.copy()
    overlay_err[wrong] = (0, 0, 255)
    img_err = cv2.addWeighted(img, 1 - alpha, overlay_err, alpha, 0)

    # 3) Combinação de TP e erros
    overlay_both = base.copy()
    overlay_both[correct] = (0, 255, 0)
    overlay_both[wrong]   = (0, 0, 255)
    img_both = cv2.addWeighted(img, 1 - alpha, overlay_both, alpha, 0)

    return img_tp, img_err, img_both

def main() -> None:
    """Load image and masks, generate overlays, save and display results."""
    # Paths hard-coded
    img_path       = '/home/rodrigo/JoseBras/art_wsl/img.png'
    gt_mask_path   = '/home/rodrigo/JoseBras/art_wsl/weak_mask.png'
    pred_mask_path = '/home/rodrigo/JoseBras/art_wsl/pred_weak.png'

    # Carrega ficheiros
    img       = cv2.imread(img_path)                                 # BGR
    gt_mask   = cv2.imread(gt_mask_path,   cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

    # Verifica leituras
    if img is None or gt_mask is None or pred_mask is None:
        raise FileNotFoundError(
            "Verifique os caminhos de 'img.png', 'weak_mask.png' e 'pred_weak.png'."
        )

    # Ajusta dimensões das máscaras à imagem
    if gt_mask.shape != img.shape[:2]:
        gt_mask = cv2.resize(
            gt_mask,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    if pred_mask.shape != img.shape[:2]:
        pred_mask = cv2.resize(
            pred_mask,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Gera as três imagens sobrepostas
    img_tp, img_err, img_both = overlay_masks(img, gt_mask, pred_mask, alpha=0.6)

    # Cria pasta de saída
    out_dir = '/home/rodrigo/JoseBras/art_wsl/overlay'
    os.makedirs(out_dir, exist_ok=True)

    # Grava resultados
    cv2.imwrite(os.path.join(out_dir, 'only_tp.png'),     img_tp)
    cv2.imwrite(os.path.join(out_dir, 'only_errors.png'), img_err)
    cv2.imwrite(os.path.join(out_dir, 'tp_errors.png'),   img_both)
    print("Resultados gravados em:", out_dir)

    # Exibe com Matplotlib
    titles = ['Só Verdadeiros Positivos', 'Só Erros', 'TP + Erros']
    images = [img_tp, img_err, img_both]

    plt.figure(figsize=(15, 5))
    for i, (title, im) in enumerate(zip(titles, images), start=1):
        plt.subplot(1, 3, i)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # BGR→RGB para exibição
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
