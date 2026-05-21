import cv2
import numpy as np


def estimate_page_angle_deg(image: np.ndarray, max_angle: float = 15.0) -> float:
    """Szacuje globalny kąt pochylenia strony metodą Hougha na liniach poziomych."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    if np.count_nonzero(edges) < 100:
        return 0.0

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

    if lines is None or len(lines) < 5:
        return 0.0

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = float(np.degrees(theta) - 90.0)
        if abs(angle_deg) <= max_angle:
            angles.append(angle_deg)

    if len(angles) < 3:
        return 0.0

    return float(np.median(angles))


def estimate_line_angle_deg(items) -> float:
    """Regresja liniowa przez centra BB komponentów linii; zwraca kąt pochylenia w stopniach."""
    if len(items) < 3:
        return 0.0

    xs = np.array([(c["x1"] + c["x2"]) / 2.0 for c in items], dtype=np.float64)
    ys = np.array([(c["y1"] + c["y2"]) / 2.0 for c in items], dtype=np.float64)

    if np.ptp(xs) < 1e-6:
        return 0.0

    A = np.column_stack([xs, np.ones_like(xs)])
    result = np.linalg.lstsq(A, ys, rcond=None)
    slope = float(result[0][0])
    return float(np.degrees(np.arctan(slope)))


def estimate_word_angle_deg(gray: np.ndarray, max_angle: float = 15.0, step: float = 0.5) -> float:
    """Szacuje kąt korekcji pochylenia wyrazu przez maksymalizację wariancji projekcji poziomej.

    Zwraca kąt do bezpośredniego zastosowania w rotate_image (kąt korekcji, nie kąt pochylenia).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(binary) < 30:
        return 0.0

    best_angle, best_var = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + step, step):
        rotated = rotate_image(binary, float(a))
        proj = np.sum(rotated > 0, axis=1).astype(np.float32)
        var = float(np.var(proj))
        if var > best_var:
            best_var, best_angle = var, float(a)

    return best_angle


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Obrót obrazu o zadany kąt z BORDER_REPLICATE (bez czarnych krawędzi)."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
