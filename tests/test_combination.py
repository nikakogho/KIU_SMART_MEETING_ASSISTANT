import pytest
from unittest.mock import MagicMock
from pyannote.core import Segment, Annotation
from services import combine_transcript_and_diarize_results

class TranscriptionWord:
    """A simple class to represent a word in the transcription."""
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end

# Fixture to create a mock Whisper transcription object
@pytest.fixture
def mock_transcription():
    # This simulates the structure of the object returned by the Whisper API
    transcription = MagicMock()
    
    transcription.words = [
        TranscriptionWord('Hello', 0.5, 0.9),
        TranscriptionWord('Nika,', 1.0, 1.3),
        TranscriptionWord('how', 2.1, 2.4),
        TranscriptionWord('are', 2.4, 2.6),
        TranscriptionWord('you?', 2.6, 2.9),
        TranscriptionWord('This', 5.5, 5.8), # This word will be skipped (no speaker)
    ]
    return transcription

# Fixture to create a mock Pyannote diarization object
@pytest.fixture
def mock_diarization():
    # This simulates the structure of the object returned by Pyannote
    annotation = Annotation()
    annotation[Segment(0, 2)] = 'SPEAKER_00'
    annotation[Segment(2, 4)] = 'SPEAKER_01'
    return annotation

def test_main_success_flow(mock_diarization, mock_transcription):
    """Tests that words are correctly assigned to speakers in the right order."""
    expected_output = "[00:00] SPEAKER_00: Hello Nika,\n[00:02] SPEAKER_01: how are you?"
    result = combine_transcript_and_diarize_results(mock_diarization, mock_transcription)
    assert result == expected_output

def test_edge_case_no_words_in_segment(mock_transcription):
    """Tests a speaker turn that has no corresponding words."""
    annotation = Annotation()
    annotation[Segment(0, 2)] = 'SPEAKER_00'
    annotation[Segment(8, 10)] = 'SPEAKER_01' # This segment has no words from the mock
    
    expected_output = "[00:00] SPEAKER_00: Hello Nika,\n[00:08] SPEAKER_01:"
    result = combine_transcript_and_diarize_results(annotation, mock_transcription)
    assert result == expected_output

def test_failure_flow_empty_inputs():
    """Tests that the function handles empty inputs gracefully."""
    empty_transcription = MagicMock()
    empty_transcription.words = []
    empty_diarization = Annotation()
    
    expected_output = ""
    result = combine_transcript_and_diarize_results(empty_diarization, empty_transcription)
    assert result == expected_output