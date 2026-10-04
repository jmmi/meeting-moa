import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'worker'))
from engine import align_segments, clean_text, speaker_at

class AlignmentTests(unittest.TestCase):
    def test_zero_duration_words_are_not_lost(self):
        data=[{'text':' 네, 제가 하겠습니다.', 'offsets':{'from':1000,'to':2000},'tokens':[
            {'text':'[_BEG_]','offsets':{'from':1000,'to':1000}},
            {'text':' 네, 제가','offsets':{'from':1000,'to':1000}},
            {'text':' 하겠습니다.','offsets':{'from':1000,'to':2000}}]}]
        r=align_segments(data,[{'start':1,'end':2,'speaker':'a'}],3)
        self.assertEqual(r[0]['text'],'네, 제가 하겠습니다.')
    def test_split_one_sentence_at_speaker_change(self):
        data=[{'text':' Hello. Yes.', 'offsets':{'from':0,'to':3000},'tokens':[
            {'text':' Hello.','offsets':{'from':0,'to':1000}},
            {'text':' Yes.','offsets':{'from':2000,'to':3000}}]}]
        r=align_segments(data,[{'start':0,'end':1,'speaker':'a'},{'start':2,'end':3,'speaker':'b'}],3)
        self.assertEqual([(s['speaker'],s['text']) for s in r],[('a','Hello.'),('b','Yes.')])
    def test_unknown_gap_is_not_invented_speaker(self):
        self.assertEqual(speaker_at(4,5,[{'start':0,'end':1,'speaker':'a'}]),'unknown')
    def test_bad_token_decoding_preserves_full_segment(self):
        r=align_segments([{'text':'안녕하세요','offsets':{'from':0,'to':1000},'tokens':[{'text':'잘못된 토큰','offsets':{'from':0,'to':1000}}]}],[{'start':0,'end':1,'speaker':'a'}],1)
        self.assertEqual(r[0]['text'],'안녕하세요')
    def test_speaker_change_never_splits_a_word(self):
        data=[{'text':' 그쵸 네','offsets':{'from':0,'to':3000},'tokens':[
            {'text':' 그','offsets':{'from':0,'to':900}},
            {'text':'쵸','offsets':{'from':1100,'to':1500}},
            {'text':' 네','offsets':{'from':2000,'to':3000}}]}]
        r=align_segments(data,[{'start':0,'end':1,'speaker':'a'},{'start':1,'end':1.6,'speaker':'b'},{'start':2,'end':3,'speaker':'b'}],3)
        self.assertEqual([(s['speaker'],s['text']) for s in r],[('a','그쵸'),('b','네')])
    def test_repeated_lines_are_dropped(self):
        line=lambda t:{'text':' 아 엄청 오래 하셨구나.','offsets':{'from':t,'to':t+1000},'tokens':[]}
        r=align_segments([line(0),line(1000),line(2000)],[{'start':0,'end':3,'speaker':'a'}],3)
        self.assertEqual([s['text'] for s in r],['아 엄청 오래 하셨구나.'])
    def test_clean_text_removes_loops_and_dashes(self):
        self.assertEqual(clean_text('-네. - 저기 관리. 저기 관리. 저기 관리. 저기 관리. 끝'),'네. 저기 관리. 끝')
        self.assertEqual(clean_text('중독독독독독독독대는'),'중독독독대는')
        self.assertEqual(clean_text('그쵸 그쵸 그쵸 A-B -5도'),'그쵸 그쵸 그쵸 A-B -5도')
if __name__=='__main__':unittest.main()
