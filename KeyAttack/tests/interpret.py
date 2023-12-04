KEY_MAP = {
    "Backslash": "\\",
    "Period": ".",
    "ForwardSlash": "/",
    "Space": " ",
    "Enter": "\n"
}

# Map of unshifted: shifted keys
SHIFT_MAP = {
    "`": "~",
    "1": "!",
    "2": "@",
    "3": "#",
    "4": "$",
    "5": "%",
    "6": "^",
    "7": "&",
    "8": "*",
    "9": "(",
    "0": ")",
    "-": "_",
    "=": "+",
    "[": "{",
    "]": "}",
    "Backslash": "|",
    ";": ":",
    "'": '"',
    ",": "<",
    "Period": ">",
    "ForwardSlash": "?",
}


# Valid keys
VALID_KEYS = (
    *"0123456789abcdefghijklmnopqrstuvwxyz`-=[];',",
    "Backspace",
    "Backslash",
    "CapsLock",
    "Enter",
    "Period",
    "ForwardSlash",
    "Space",
    "ShiftPress",
    "ShiftRelease"
)


def interpret_seq(seq: list[str]):
    """
    Interpret a list of key presses as a string
    input: list of key presses such as ["a", "Backspace", "b", "CapsLock", "c", "CapsLock", "d", "e", "ShiftDown", "f", "1", "ShiftRelease", "g"]
    output: string such as "bCdeF!g"

    Code by Curtis Heizl and fixes by Kevin Zhang
    """
    result = []
    caps = False
    shift = False
    for i, ch in enumerate(seq):
        if ch == "Backspace" and len(result) > 0:
            result.pop()
        elif ch == "CapsLock":
            caps = not caps
        elif ch == "ShiftDown":
            shift = True
        elif ch == "ShiftRelease":
            shift = False
        elif shift and ch in SHIFT_MAP:  # numbers and symbols
            result.append(
                SHIFT_MAP[ch]
            )  # does not respond to caps lock
        elif ch in KEY_MAP:
            result.append(
                KEY_MAP[ch]
            )
        elif 97 <= ord(ch) <= 122:  # lowercase letters
            result.append(
                ch.upper() if caps ^ shift else ch
            )  # uppercase if caps XOR shift is activated
        else:  # Should only include characters whose
            result.append(ch)
    return "".join(result)


def test():
    inputs = [
        [
            "a",
            "Backspace",
            "b",
            "CapsLock",
            "c",
            "CapsLock",
            "d",
            "e",
            "ShiftDown",
            "f",
            "1",
            "ShiftRelease",
            "g",
        ],
        ["a", "CapsLock", "Backslash", "b", "ShiftDown", "c", "ShiftRelease", "d"],
        ["[", "ShiftDown", "]", "ShiftRelease"],
        [
            "CapsLock",
            "CapsLock",
            "CapsLock",
            "0",
            "a",
            "CapsLock",
            "CapsLock",
            "CapsLock",
            "0",
            "a",
        ],
    ]
    outputs = ["bCdeF!g", "a\\BcD", "[}", "0A0a"]
    for i in range(len(inputs)):
        assert interpret_seq(inputs[i]) == outputs[i]


if __name__ == '__main__':
    test()
