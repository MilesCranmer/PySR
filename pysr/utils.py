import os
import re

from sklearn.utils.validation import _check_feature_names_in


def _csv_filename_to_pkl_filename(csv_filename: str) -> str:
    if os.path.splitext(csv_filename)[1] == ".pkl":
        return csv_filename

    # Assume that the csv filename is of the form "foo.csv"
    assert str(csv_filename).endswith(".csv")

    dirname = str(os.path.dirname(csv_filename))
    basename = str(os.path.basename(csv_filename))
    base = str(os.path.splitext(basename)[0])

    pkl_basename = base + ".pkl"

    return os.path.join(dirname, pkl_basename)


_regexp_im = re.compile(r"\b(\d+\.\d+)im\b")
_regexp_im_sci = re.compile(r"\b(\d+\.\d+)[eEfF]([+-]?\d+)im\b")
_regexp_sci = re.compile(r"\b(\d+\.\d+)[eEfF]([+-]?\d+)\b")

_apply_regexp_im = lambda x: _regexp_im.sub(r"\1j", x)
_apply_regexp_im_sci = lambda x: _regexp_im_sci.sub(r"\1e\2j", x)
_apply_regexp_sci = lambda x: _regexp_sci.sub(r"\1e\2", x)


def _preprocess_julia_floats(s: str) -> str:
    if isinstance(s, str):
        s = _apply_regexp_im(s)
        s = _apply_regexp_im_sci(s)
        s = _apply_regexp_sci(s)
    return s


def _safe_check_feature_names_in(self, variable_names, generate_names=True):
    """_check_feature_names_in with compat for old versions."""
    try:
        return _check_feature_names_in(
            self, variable_names, generate_names=generate_names
        )
    except TypeError:
        return _check_feature_names_in(self, variable_names)


def _subscriptify(i: int) -> str:
    """Converts integer to subscript text form.

    For example, 123 -> "₁₂₃".
    """
    return "".join([chr(0x2080 + int(c)) for c in str(i)])
