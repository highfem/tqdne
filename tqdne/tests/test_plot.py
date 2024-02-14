import matplotlib.pyplot as plt
from tqdne.utils import fig2PIL
import PIL
import os


def test_fig2PIL():
    fig = plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.close()
    pil = fig2PIL(fig)
    # save the PIL image
    pil.save("test_fig2PIL.png")

    # load the PIL image
    pil = PIL.Image.open("test_fig2PIL.png")

    # delete the PIL image
    os.remove("test_fig2PIL.png")
