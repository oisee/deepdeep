#!/bin/bash
# Helper script to trigger interactive menu during DeepDeep search

echo "🎛️  Triggering DeepDeep interactive menu..."
touch menu.trigger
echo "✅ Menu trigger created!"
echo ""
echo "The running DeepDeep process should show the interactive menu shortly."
echo "If it doesn't appear, try: touch .pause"