# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def rescale(self, xScale, yScale, diffx, diffy):

        xStart = (self.tlwh[0] + diffx) * xScale #+ x1
        yStart = (self.tlwh[1] + diffy)  * yScale #+ y1

        width = (self.tlwh[2]) * xScale #+ x2
        height = (self.tlwh[3]) * yScale #+ y2
        self.tlwh = np.asarray([xStart, yStart, width, height], dtype=np.float32)
        return self

    def validBbox(self):
        bbox = self.to_tlbr()
        if abs(bbox[0] - bbox[2])<=1 or abs(bbox[1] - bbox[3]) <=1:
            return False
        return True