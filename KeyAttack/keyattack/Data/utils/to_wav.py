"""
Written by Edward Ng.
"""


from pydub import AudioSegment
import glob


if __name__ == "__main__":
    # load file paths
    file_extension = '*.m4a'
    files = glob.glob('FOLDER_PATH' + file_extension)

    for file in files:
        wav_filename = file.replace('.m4a', '.wav')

        # convert to wav
        sound = AudioSegment.from_file(file, format='m4a')
        file_handle = sound.export(wav_filename, format='wav')
