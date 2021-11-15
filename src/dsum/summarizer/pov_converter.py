import argparse


def convert_pov(text, speaker=None, previous_speaker=None):

    change_to_singular = False

    # 1 - Change pronouns.
    text = text.replace(" we ", " they ")
    text = text.replace(" our ", " their ")
    text = text.replace(" us ", " them ")
    text = text.replace(" me ", " him ")
    text = text.replace(" i ", " he ")
    text = text.replace("i'll", "he'll")
    text = text.replace("i've", "he has")
    text = text.replace("we've", "they've")
    text = text.replace("we'll", "they'll")
    text = text.replace("let's", "they")
    if speaker:
        text = text.replace(" mine ", f" {speaker}'s ")
    if previous_speaker:
        text = text.replace(" you ", f" {previous_speaker} ")
        text = text.replace(" your ", f" {previous_speaker}'s ")

    # 2 - Subject verb agreement.
    change_to_singular = " i " in text or " you " in text

    # 3 - Modal verbs change.
    text = text.replace(" can ", " them ")
    text = text.replace(" may ", " might ")
    text = text.replace(" must ", " has to ")

    # 4 - Convert questions.
    if text.strip()[-1] == "?" and speaker:
        text = f"{speaker} asks if {text[:-1]}"

    # 5 - Remove unnecessary words.
    text = text.replace(" ok ", "ok")

    return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument")
    args = parser.parse_args()
