import cv2
import numpy as np


class LivenessDetector:
    """
    Basic liveness detector using texture-analysis heuristics.

    Two metrics are combined:
    1. Laplacian variance  – measures sharpness / high-frequency detail.
    2. Gradient magnitude variance – measures edge richness.

    Printed photos or phone screens tend to score low on both.
    """

    def __init__(self, threshold=60.0):
        self.threshold = threshold

    def _laplacian_score(self, gray: np.ndarray) -> float:
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _gradient_score(self, gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.var(magnitude))

    def is_live(self, face_img: np.ndarray):
        """
        Determine whether the presented face is likely live.

        Parameters
        ----------
        face_img : np.ndarray
            BGR face crop from OpenCV.

        Returns
        -------
        (bool, str)
            is_live flag and a descriptive message always returned as a tuple.
        """
        if face_img is None or face_img.size == 0:
            return False, "No Face"

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        lap_score = self._laplacian_score(gray)
        grad_score = self._gradient_score(gray)

        # Combined score (weighted average; weights tunable)
        combined = 0.6 * lap_score + 0.4 * (grad_score / 100.0)

        if combined < self.threshold:
            return False, f"Spoof ({combined:.1f})"

        return True, f"Live ({combined:.1f})"
