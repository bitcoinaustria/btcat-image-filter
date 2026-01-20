import unittest.mock as mock
import pytest
from textual.widgets import Label, ListItem
from textual.app import App
import sys
from pathlib import Path

# Add project root to path so we can import tui
sys.path.append(str(Path(__file__).parent.parent))

import tui

class TestFileSelection:
    @pytest.fixture
    def screen(self):
        """Fixture to create a FileSelectionScreen with a mocked app."""
        screen = tui.FileSelectionScreen()
        # Mock the app property using patch.object on the instance would be tricky
        # because app is a property on the Screen class.
        # Instead, we can assign a mock to the _app attribute if Textual uses that,
        # or mock the property on the class for the duration of the test.
        return screen

    def test_on_list_view_selected(self):
        """Test that selecting a file item triggers the push_screen method."""
        
        # Setup the screen and mock app
        screen = tui.FileSelectionScreen()
        mock_app = mock.MagicMock()
        
        # We need to mock the 'app' property. Since it's a property on the base class,
        # we can patch it on FileSelectionScreen.
        with mock.patch.object(tui.FileSelectionScreen, 'app', new_callable=mock.PropertyMock) as mock_app_prop:
            mock_app_prop.return_value = mock_app
            
            # Create a mock event
            event = mock.MagicMock()
            
            # Create a fake label and list item
            filename = "test_image.jpg"
            label = Label(filename)
            
            # Mock query_one to return our label
            # We assume event.item is the ListItem
            event.item.query_one.return_value = label
            
            # Also need to ensure DitheringScreen doesn't crash or tries to load the file
            # tui.DitheringScreen checks if file exists.
            with mock.patch('pathlib.Path.exists', return_value=True):
                # Also mock Image.open to avoid actual file I/O
                with mock.patch('PIL.Image.open') as mock_open:
                    mock_image = mock.MagicMock()
                    mock_image.mode = 'RGB'
                    mock_open.return_value = mock_image
                    
                    # Call the method
                    screen.on_list_view_selected(event)
                    
            # Verify that push_screen was called
            assert mock_app.push_screen.called
            
            # Get the args passed to push_screen
            args, _ = mock_app.push_screen.call_args
            pushed_screen = args[0]
            
            assert isinstance(pushed_screen, tui.DitheringScreen)
            assert pushed_screen.image_path.name == filename

    def test_on_list_view_selected_no_files(self):
        """Test that selecting 'No image files...' does nothing."""
         # Setup the screen
        screen = tui.FileSelectionScreen()
        
        with mock.patch.object(tui.FileSelectionScreen, 'app', new_callable=mock.PropertyMock) as mock_app_prop:
            mock_app = mock.MagicMock()
            mock_app_prop.return_value = mock_app
            
            event = mock.MagicMock()
            label = Label("No image files found in current directory")
            event.item.query_one.return_value = label
            
            screen.on_list_view_selected(event)
            
            # verify push_screen was NOT called
            assert not mock_app.push_screen.called
