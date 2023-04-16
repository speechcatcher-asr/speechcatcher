import jiwer
from somajo import SoMaJo
from jiwer import cer, wer, wil
import argparse

punctuation = '.,!?;:-_"\''

def to_word_list(tokenizer, paragraphs, remove_punctuation=False):

    paragraph_ids = list(sorted(paragraphs.keys()))
    paragraphs = [paragraphs[key] for key in paragraph_ids]

    if remove_punctuation: 
        paragraph_tokens = [[token.text for token in sentence if token.text not in punctuation] for sentence in tokenizer.tokenize_text(paragraphs)]
    else:
        paragraph_tokens = [[token.text for token in sentence] for sentence in tokenizer.tokenize_text(paragraphs)]


    assert(len(paragraph_ids) == len(paragraph_tokens))

    return dict(zip(paragraph_ids, paragraph_tokens))

def calculate_word_error_rate(kaldi_test_text_file, decoded_text_file, remove_punctuation=False, split_camel_case=True, lower_case=False):
    tokenizer = SoMaJo(language="de_CMC", split_camel_case=split_camel_case, split_sentences=False)

    test_utterances = {}
    decoded_utterances = {}

    with open(kaldi_test_text_file) as kaldi_test_text_file_in:
        for line in kaldi_test_text_file_in:
            utterance_id, transcription = line.strip().split(' ', 1)
            test_utterances[utterance_id] = transcription

    with open(decoded_text_file) as decoded_text_file_in:
        for line in decoded_text_file_in:
            utterance_id, transcription = line.strip().split(' ', 1)
            decoded_utterances[utterance_id] = transcription

    test_utterances = to_word_list(tokenizer, test_utterances, remove_punctuation)
    decoded_utterances = to_word_list(tokenizer, decoded_utterances, remove_punctuation)
    decoded_utterances_ids = list(sorted(decoded_utterances.keys()))

    if lower_case:
        test_utterances_list = [' '.join(test_utterances[utt_id]).lower() for utt_id in decoded_utterances_ids]
        decoded_utterances_list = [' '.join(decoded_utterances[utt_id]).lower() for utt_id in decoded_utterances_ids]
    else:
        test_utterances_list = [' '.join(test_utterances[utt_id]) for utt_id in decoded_utterances_ids]
        decoded_utterances_list = [' '.join(decoded_utterances[utt_id]) for utt_id in decoded_utterances_ids]


    print('CER:', cer(test_utterances_list, decoded_utterances_list))
    print('WER:', wer(test_utterances_list, decoded_utterances_list))
    print('WIL:', wil(test_utterances_list, decoded_utterances_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--remove_punctuation', dest='remove_punctuation',
                        help='Remove punctuation before calculating metrics', action='store_true')
    parser.add_argument('--lower_case', dest='lower_case',
                        help='Lower case all word before calculating metrics', action='store_true')

    parser.add_argument('testset_text', type=str, help='The test file containing the test set utterances')
    parser.add_argument('decoded_text', type=str, help='The decoded file containing the decoded utterances')

    args = parser.parse_args()

    calculate_word_error_rate(args.testset_text, args.decoded_text, remove_punctuation=args.remove_punctuation, lower_case=args.lower_case)
