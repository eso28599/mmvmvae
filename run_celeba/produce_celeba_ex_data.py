import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import grid
from matplotlib.patches import Patch

with open("utils/alphabet.json") as alphabet_file:
    alphabet_ex = str("".join(json.load(alphabet_file)))

def char2Index(alphabet, character):
    return alphabet.find(character)
  
def one_hot_encode(len_seq, alphabet, seq):
    """Returns a one-hot encoded tensor of shape (length of sequence, length of alphabet) for the given sequence and alphabet."""
    # length of the sequence (including padding) x length of the alphabet
    X = np.zeros((len_seq, len(alphabet)))
    if len(seq) > len_seq:
        seq = seq[:len_seq]
    # X_ij = 1 if the i-th character in the sequence is the 
    # j-th character in the alphabet, 0 otherwise
    # if the sequence is shorter than the maximum length
    # the corresponding rows will be all zeros (padding)
    for index_char, char in enumerate(seq):
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1.0
    return X
  
seq_ex = "attractive, male, mouth slightly open, no beard, smiling"
terms = seq_ex.split(", ")
random_start = 33
padded_seq = random_start * '*' + seq_ex + (256 - len(seq_ex) - random_start) * '*'
example_encode = one_hot_encode(256, alphabet_ex, padded_seq)

for full in [True, False]:
  start_square = 0 if full else random_start - 4
  end_square = 256 if full else random_start + len(seq_ex) + 4
  for grid in [True, False]:
    if full & grid:
      continue
    ex_encode = example_encode[start_square:end_square, :] 
    # Assign unique values and collect start/end indices
    term_indices = []
    yticks = []
    yticklabels = []
    yticks_indices = []
    for i, term in enumerate(terms):
        start = seq_ex.find(term) 
        end = start + len(term) -1
        term_indices.append((start + 4-0.5, end+4 + 0.5, term))
        yticks.extend([start + 4 -0.5, end + 4 +0.5])
        yticklabels.extend([f"'{term}' starts", f"'{term}' ends"])
        yticks_indices.extend([start + 1 + 4, end +1 + 4])

    # Remove duplicates and keep order
    ytick_tuples = list(dict.fromkeys(zip(yticks, yticklabels)))
    yticks, yticklabels = zip(*ytick_tuples)

    # Set zeros to NaN for visualization
    ex_encode[ex_encode == 0] = np.nan

    # Plot
    if full:
      fig, ax = plt.subplots(figsize=(2.2, 8))
      im = ax.imshow(ex_encode, cmap='viridis')
      ax.set_ylabel('Row index $i$', fontsize=14)
      ax.set_xlabel('Column index $j$', fontsize=14)
    else:
      fig, ax = plt.subplots()
      im = ax.imshow(ex_encode, cmap='viridis', aspect='auto')
      # Original y-axis (left)
      ax.set_ylabel('Sequence Position (row index $i$)')
      ax.set_xlabel('Alphabet Index (column index $j$)')

    # Create legend
    colors = [plt.cm.viridis((i+5)/ (len(terms)+5)) for i in range(len(terms))]
    legend_handles = [Patch(color=colors[i], label=terms[i]) for i in range(len(terms))]
    if not full:
      for start, end, term in term_indices:
          plt.axhline(y=start, color = 'black',linestyle='--', linewidth=0.5)
          plt.axhline(y=end, color = 'black', linestyle='--', linewidth=0.5)
      plt.autoscale(False)
    ax.set_ylim(ex_encode.shape[0]-0.5, -0.5)

    if grid:
      plt.grid(linewidth=0.5, alpha=0.7)
      ax.set_yticks([i - 0.5 for i in range(0, 64)])
      ax.set_yticklabels(
        [i + random_start - 4 if i+1 in yticks_indices else "" for i in range(0, 64)]
        )
      ax.set_xticks([i - 0.5 for i in range(0, 42)])
      ax.set_xticklabels(
        [i if i in range(0, 42, 5) else "" for i in range(0, 42)]
        )
      filename = "run_celeba/text_example_submatrix_grid.png"
    elif not full:
      ax.set_yticks(yticks)
      ax.set_yticklabels([index + random_start - 5 for index in yticks_indices])
      ax.set_xticks([i-0.5 for i in range(0, 42, 5)])
      ax.set_xticklabels([i for i in range(0, 42, 5)])
      filename = "run_celeba/text_example_submatrix.png"
    else:
      ax.tick_params(labelsize=14)
      ax.set_xticks([i-0.5 for i in range(0, 42, 20)])
      ax.set_xticklabels([i for i in range(0, 42, 20)])
      ax.set_yticks([i-0.5 for i in range(0, 256, 50)])
      ax.set_yticklabels([i for i in range(0, 256, 50)])
      filename = "run_celeba/text_example_full_matrix.png"

    # Twin y-axis for custom ticks (right)
    if not full:
      ax_right = ax.twinx()
      ax_right.set_ylim(ax.get_ylim())  # Match left axis limits
      ax_right.set_yticks(yticks)
      ax_right.set_yticklabels(yticklabels)
      ax_right.set_ylim(ex_encode.shape[0] - 0.5, -0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)