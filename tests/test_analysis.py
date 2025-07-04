# tests/test_analysis.py
import pytest
from unittest.mock import MagicMock
from services import analyze_meeting_transcript

# A fixture to create a reusable mock OpenAI client
@pytest.fixture
def mock_openai_client(mocker): # 'mocker' is provided by pytest-mock
    """Mocks the OpenAI client and its API calls."""
    mock_client = MagicMock()
    
    # This is the fake JSON response we want the API to "return"
    fake_json_response = '{"summary": "A test summary.", "decisions_made": ["Test decision"], "action_items": [], "effectiveness_rating": 8}'
    
    # Configure the mock object to simulate the nested structure of the real response
    mock_response_object = MagicMock()
    mock_response_object.choices = [MagicMock()]
    mock_response_object.choices[0].message = MagicMock()
    mock_response_object.choices[0].message.content = fake_json_response
    
    # Tell the mock client that when 'chat.completions.create' is called, it should return our fake response
    mock_client.chat.completions.create.return_value = mock_response_object
    
    return mock_client

def test_main_success_flow(mock_openai_client):
    """Tests that a valid transcript produces a valid parsed dictionary."""
    transcript = "This is a test transcript."
    result = analyze_meeting_transcript(transcript, mock_openai_client)
    
    # Assert that the function correctly parses the mocked JSON response
    assert isinstance(result, dict)
    assert result["summary"] == "A test summary."
    assert result["effectiveness_rating"] == 8

def test_edge_case_empty_transcript(mock_openai_client):
    """Tests that the function does not call the API if the transcript is empty."""
    # We are assuming the real function has a check like `if not transcript_text: return None`
    # For this test to pass, you might need to add that check to your function.
    transcript = ""
    result = analyze_meeting_transcript(transcript, mock_openai_client)
    
    # Assert that the function returns None or an empty dict
    assert result is None
    # Assert that the expensive API call was never made
    mock_openai_client.chat.completions.create.assert_not_called()

def test_failure_flow_api_error(mock_openai_client):
    """Tests how the function handles an exception from the OpenAI API."""
    # Configure the mock to raise an exception instead of returning a value
    mock_openai_client.chat.completions.create.side_effect = Exception("API connection timed out")
    
    transcript = "A valid transcript that will cause a mocked error."
    result = analyze_meeting_transcript(transcript, mock_openai_client)
    
    # The function should catch the exception and return None
    assert result is None