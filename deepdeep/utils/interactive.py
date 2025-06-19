"""
Interactive utilities for user input during search.
"""

import sys
import select
import termios
import tty
from typing import Optional, Dict, Any
from enum import Enum

class InteractiveChoice(Enum):
    CONTINUE = "continue"
    SAVE_INTERMEDIATE = "save"
    EXIT = "exit"
    SKIP_PHASE = "skip"

class InteractiveHandler:
    """Handle interactive input during search operations."""
    
    def __init__(self):
        self.original_settings = None
        self.enabled = True
        
    def setup_non_blocking_input(self):
        """Setup terminal for non-blocking input."""
        if not sys.stdin.isatty():
            self.enabled = False
            return
            
        try:
            self.original_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        except:
            self.enabled = False
    
    def restore_input(self):
        """Restore terminal settings."""
        if self.original_settings and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)
            except:
                pass
    
    def check_for_interactive_trigger(self) -> Optional[InteractiveChoice]:
        """Check for interactive trigger (file-based or key-based)."""
        import os
        
        # Check for trigger file (more reliable)
        if os.path.exists('menu.trigger'):
            os.remove('menu.trigger')  # Remove trigger file
            return self._show_interactive_menu()
            
        if not self.enabled or not sys.stdin.isatty():
            return None
            
        # Check if input is available (non-blocking)
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            try:
                # Simple single character check
                char = sys.stdin.read(1)
                # Check for common trigger characters
                if char.lower() in ['m', 'i', '?']:  # m=menu, i=interactive, ?=help
                    return self._show_interactive_menu()
            except:
                pass
        
        return None
    
    def _show_interactive_menu(self) -> InteractiveChoice:
        """Show interactive menu and get user choice."""
        # Restore terminal for menu interaction
        self.restore_input()
        
        print("\n" + "="*50)
        print("🎛️  INTERACTIVE MENU")
        print("="*50)
        print("Choose an action:")
        print("  [c] Continue search")
        print("  [s] Save intermediate results and continue")
        print("  [x] Exit and save current best results")
        print("  [k] Skip current phase")
        print("="*50)
        print("💡 Tip: During search, create file 'menu.trigger' to open this menu")
        
        while True:
            try:
                choice = input("Your choice [c/s/x/k]: ").lower().strip()
                
                if choice in ['c', 'continue', '']:
                    print("▶️  Continuing search...")
                    self.setup_non_blocking_input()
                    return InteractiveChoice.CONTINUE
                    
                elif choice in ['s', 'save']:
                    print("💾 Saving intermediate results and continuing...")
                    self.setup_non_blocking_input()
                    return InteractiveChoice.SAVE_INTERMEDIATE
                    
                elif choice in ['x', 'exit']:
                    print("🚪 Exiting and saving current best results...")
                    return InteractiveChoice.EXIT
                    
                elif choice in ['k', 'skip']:
                    print("⏭️  Skipping current phase...")
                    self.setup_non_blocking_input()
                    return InteractiveChoice.SKIP_PHASE
                    
                else:
                    print(f"Invalid choice: '{choice}'. Please choose c/s/x/k")
                    
            except KeyboardInterrupt:
                print("\n🚪 Ctrl+C detected - exiting...")
                return InteractiveChoice.EXIT
            except EOFError:
                print("\n🚪 EOF detected - exiting...")
                return InteractiveChoice.EXIT
    
    def __enter__(self):
        """Context manager entry."""
        self.setup_non_blocking_input()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.restore_input()


def create_search_config_with_interactive(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create search config with interactive handler enabled."""
    config = base_config.copy()
    config['interactive_handler'] = InteractiveHandler()
    return config