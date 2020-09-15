EPS = "<eps>"  # Define once. Use the same EPS everywhere

CHARS = list("abcdefghijklmnopqrstuvwxyz")

INFINITY = 1000000000


def calculate_arc_weight(frequency):
    """Function to calculate the weight of an arc based on a frequency count

    Args:
        frequency (float): Frequency count

    Returns:
        (float) negative log of frequency

    """
    # TODO: INSERT YOUR CODE HERE
    raise NotImplementedError(
        "You need to implement calculate_arc_weight function in scripts/util.py!!!"
    )


def format_arc(src, dst, ilabel, olabel, weight=0):
    """Create an Arc, i.e. a line of an openfst text format file

    Args:
        src (int): source state
        dst (int): sestination state
        ilabel (str): input label
        olabel (str): output label
        weight (float): arc weight

    Returns:
        (str) The formatted line as a string
    http://www.openfst.org/twiki/bin/view/FST/FstQuickTour#CreatingShellFsts
    """
    # TODO: INSERT YOUR CODE HERE
    raise NotImplementedError(
        "You need to implement format_arc function in scripts/util.py!!!"
    )
