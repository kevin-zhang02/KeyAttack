KEY_MAP = {
    "Backslash": "\\",
    "Period": ".",
    "ForwardSlash": "/",
    "Space": " ",
    "Enter": "\n",
}

KEY_MAP_REVERSED = {v: k for k, v in KEY_MAP.items()}

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

SHIFT_MAP_REVERSED = {v: k for k, v in SHIFT_MAP.items()}


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
    "ShiftRelease",
)


def interpret_seq(seq: list[str]):
    """
    Interpret a list of key presses as a string
    input: list of key presses such as ["a", "Backspace", "b", "CapsLock", "c", "CapsLock", "d", "e", "ShiftDown", "f", "1", "ShiftRelease", "g"]
    output: string such as "bCdeF!g"

    Code by Curtis Heinzl and fixes by Kevin Zhang
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
            result.append(SHIFT_MAP[ch])  # does not respond to caps lock
        elif ch in KEY_MAP:
            result.append(KEY_MAP[ch])
        elif 97 <= ord(ch) <= 122:  # lowercase letters
            result.append(
                ch.upper() if caps ^ shift else ch
            )  # uppercase if caps XOR shift is activated
        else:  # Should only include characters whose
            result.append(ch)
    return "".join(result)


def listify_string(string: str):
    """
    Converts a string into a list of key presses
    Uses Shift for all uppercase and special characters
    input: string such as "bCdeF!g"
    output: list of key presses such as ["b", "ShiftDown", "c", "ShiftRelease", "d", "e", "ShiftDown", "f", "1", "ShiftRelease", "g"]
    """
    result = []
    shift = False
    for ch in string:
        if ch in KEY_MAP_REVERSED:  # turn special characters into the name of the key
            ch = KEY_MAP_REVERSED[ch]
        if ch.isupper() or ch in SHIFT_MAP.values():  # shift should be down
            if not shift:
                result.append("ShiftDown")
                shift = True
        else:  # shift should be up
            if shift:
                result.append("ShiftRelease")
                shift = False

        # turn shifted characters into unshifted characters
        if ch in SHIFT_MAP_REVERSED.keys():
            ch = SHIFT_MAP_REVERSED[ch]
        elif len(ch) == 1:
            ch = ch.lower()

        result.append(ch)

    if shift:  # release shift at end of string
        result.append("ShiftRelease")
    return result


def test_interpret_seq():
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
            *"f1",
            "ShiftRelease",
            "g",
        ],
        ["a", "CapsLock", "Backslash", "b", "ShiftDown", "c", "ShiftRelease", "d"],
        ["[", "ShiftDown", "]", "ShiftRelease"],
        [
            "CapsLock",
            "CapsLock",
            "CapsLock",
            *"0a",
            "CapsLock",
            "CapsLock",
            "CapsLock",
            *"0a",
        ],
        [
            "ShiftDown",
            *"92l",
            "ShiftRelease",
            *"azy",
            "ShiftDown",
            "d",
            "ShiftRelease",
            *"og",
            "ShiftDown",
            "-",
            "ShiftRelease",
            *"85's",
            "Space",
            *"post",
            "ShiftDown",
            "0",
            "ShiftRelease",
            "Space",
            "ShiftDown",
            "t",
            "ShiftRelease",
            *"he",
            "Space",
            *"quick,",
            "Space",
            *"brown",
            "Space",
            *"fox",
            "Space",
            *"jumps",
            "Space",
            *"over",
            "Space",
            *"13",
            "Space",
            *"lazy",
            "Space",
            *"dogs",
            "Space",
            *"at",
            "Space",
            "7",
            "ShiftDown",
            ";",
            "ShiftRelease",
            *"03",
            "Space",
            "ShiftDown",
            *"pm",
            "ShiftRelease",
            ";",
            "Space",
            *"however,",
            "Space",
            *"it's",
            "Space",
            *"too",
            "Space",
            "ShiftDown",
            "3",
            "ShiftRelease",
            "CapsLock",
            *"tired",
            "CapsLock",
            "Space",
            *"to",
            "Space",
            *"continue",
            "ShiftDown",
            "1",
            "ShiftRelease",
        ],
    ]
    outputs = [
        "bCdeF!g",
        "a\\BcD",
        "[}",
        "0A0a",
        "(@LazyDog_85's post) The quick, brown fox jumps over 13 lazy dogs at 7:03 PM; however, it's too #TIRED to continue!",
    ]
    for i in range(len(inputs)):
        assert interpret_seq(inputs[i]) == outputs[i]


def test_listify_string():
    inputs = [
        # "bCdeF!g",
        "(@LazyDog_85's post) The quick, brown fox jumps over 13 lazy dogs at 7:03 PM; however, it's too #TIRED to continue!"
        # "Amazingly, @ 3:45 pm, Dr. Zhao exclaimed, 'E=mc^2 is REVOLUTIONARY!' & promptly emailed the news to 100+ colleagues from his MacBook Pro, adding a smiley :-) and a hashtag #PhysicsBREAKTHROUGH.",
    ]
    for string in inputs:
        l = listify_string(string)
        print(f"{l} (length {len(l)})")


if __name__ == "__main__":
    test_interpret_seq()
    test_listify_string()
