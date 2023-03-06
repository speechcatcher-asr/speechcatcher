from kaldiio import ReadHelper
import os
import sys
import speechcatcher

if __name__ == '__main__':
    testset_dir = "data/tuda_raw_test/" 

    short_tag = 'de_streaming_transformer_m'
    speech2text = speechcatcher.load_model(speechcatcher.tags[short_tag])

    with open(testset_dir + f'/{short_tag}_decoded', 'w') as testset_dir_decoded:
        with ReadHelper(f'scp,p:{os.path.join(testset_dir, "wav.scp")}') as reader:
            for key, (rate, speech) in reader:
                print(key, min(speech), max(speech), len(speech))
                try:
                    text = speechcatcher.recognize(speech2text, speech, rate, quiet=True, progress=False)
                    print(f'{key} {text}')
                    testset_dir_decoded.write(f'{key} {text}\n')
                except:
                    print('Warning: couldnt decode:', key)
