#!/usr/bin/env zsh

# Open the text file for reading
exec 3< "test_headlines"

# Loop through each line of the file
while IFS= read -r line <&3; do
  # Process the line
  echo "Processing line: $line"
  pyenv exec python is_sports.py "$line"
done

# Close the file descriptor
exec 3<&-
