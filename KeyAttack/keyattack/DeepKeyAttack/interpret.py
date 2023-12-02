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
    "\\": "|",
    ";": ":",
    "'": '"',
    ",": "<",
    ".": ">",
    "/": "?",
}


def interpret_seq(seq: list[str]):
    """
    Interpret a list of key presses as a string
    input: list of key presses such as ["a", "Backspace", "b", "Caps Lock", "c", "Caps Lock", "d", "e", "Shift Down", "f", "1", "Shift Release", "g"]
    output: string such as "bCdeF!g"
    """
    result = []
    caps = False
    shift = False
    for i, ch in enumerate(seq):
        if ch == "Backspace" and len(result) > 0:
            result.pop()
        elif ch == "Caps Lock":
            caps = not caps
        elif ch == "Shift Down":
            shift = True
        elif ch == "Shift Release":
            shift = False
        elif ord(ch) >= 97 and ord(ch) <= 122:  # lowercase letters
            result.append(
                ch.upper() if (caps ^ shift) else ch
            )  # uppercase if caps XOR shift is activated
        elif ch in SHIFT_MAP:  # numbers and symbols
            result.append(
                SHIFT_MAP[ch] if shift else ch
            )  # does not respond to caps lock
        elif ch == "Space":
            result.append(" ")
        elif ch == "Enter" or ch == "Tab":
            result.append("\n")
    return "".join(result)


def test():
    inputs = [
        [
            "a",
            "Backspace",
            "b",
            "Caps Lock",
            "c",
            "Caps Lock",
            "d",
            "e",
            "Shift Down",
            "f",
            "1",
            "Shift Release",
            "g",
        ],
        ["a", "Caps Lock", "\\", "b", "Shift Down", "c", "Shift Release", "d"],
        ["[", "Shift Down", "]", "Shift Release"],
        [
            "Caps Lock",
            "Caps Lock",
            "Caps Lock",
            "0",
            "a",
            "Caps Lock",
            "Caps Lock",
            "Caps Lock",
            "0",
            "a",
        ],
    ]
    outputs = ["bCdeF!g", "a\BcD", "[}", "0A0a"]
    for i in range(len(inputs)):
        assert interpret_seq(inputs[i]) == outputs[i]
