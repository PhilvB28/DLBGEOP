import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class
class CRISPRDataset(Dataset): #works only for 1DCNN
    def __init__(self, df, sequence_length=60):
        self.sequences = df['target_seq'].tolist()

        #self.labels = np.sum(df[['-1+1', '0+1', '1+1']].values.astype(np.float32), axis=1)
        self.labels = df.iloc[:, -557:].values.astype(np.float32)
        #self.labels = np.sum(df.iloc[:, -557:].values.astype(np.float32), axis=1)

        #self.labels = df.iloc[:, -557:].values.astype(np.float32)  # General Prediction
        #self.labels_deletion = df.iloc[:, 2:-21].values.astype(np.float32)  # Deletion Frequency
        #self.labels_1bp_deletion = df[['-1+1', '0+1', '1+1']].values.astype(np.float32)  # 1bp Deletion
        #self.labels_1bp_insertion = df.iloc[:, -21:-17].values.astype(np.float32)  # 1bp Insertion

        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # One-hot encode
        onehot = dna_to_onehot_extra([sequence], self.sequence_length) #squeeze?
        #onehot = dna_to_onehot([sequence], self.sequence_length)  # squeeze?

        return onehot, torch.tensor(label, dtype=torch.float32, device=device) #VORHER label

#not used at the moment
class CRISPRDatasetMT(Dataset): #for Multitask
    def __init__(self, df, sequence_length=60):
        self.sequences = df['target_seq'].tolist()
        self.sequence_length = sequence_length

        self.labels_general = df.iloc[:, -557:].values.astype(np.float32)  # General Prediction
        self.labels_deletion = df.iloc[:, 2:-21].values.astype(np.float32)  # Deletion Frequency    #TODO now correct?
        self.labels_1bp_deletion = df[['-1+1', '0+1', '1+1']].values.astype(np.float32)  # 1bp Deletion
        self.labels_1bp_insertion = df.iloc[:, -21:-17].values.astype(np.float32)  # 1bp Insertion

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # One-hot encode
        onehot = dna_to_onehot([sequence], self.sequence_length)  # Shape: (1, 4, 60)
        onehot = onehot.squeeze(0)  # Shape: (4, 60)

        # Retrieve labels for each task
        label_general = torch.tensor(self.labels_general[idx], dtype=torch.float32, device=device)
        label_deletion = torch.tensor(self.labels_deletion[idx], dtype=torch.float32, device=device)
        label_1bp_deletion = torch.tensor(self.labels_1bp_deletion[idx], dtype=torch.float32, device=device)
        label_1bp_insertion = torch.tensor(self.labels_1bp_insertion[idx], dtype=torch.float32, device=device)

        return onehot, {
            'general': label_general,
            'deletion': label_deletion,
            '1bp_deletion': label_1bp_deletion,
            '1bp_insertion': label_1bp_insertion
        }


def find_microhomologies(sequence):
    """
    Finds microhomologies around a CRISPR cut site in a 60nt DNA sequence.

    Args:
        sequence (str): A 60nt DNA sequence where the cut site is at position 30.

    Returns:
        list: A list of microhomologous sequences.
    """
    if len(sequence) != 60:
        raise ValueError("The input DNA sequence must be exactly 60 nucleotides long.")

    # Initialize variables
    cut_site = 30
    left = sequence[:cut_site]  # (positions 0 to 29)
    right = sequence[cut_site:]  # (positions 30 to 59)
    microhomologies = []

    # Find microhomologies
    max_length = min(len(left), len(right))  # Maximum length for microhomologies
    for i in range(max_length):
        if left[-(i + 1):] == right[:i + 1]:  # Check for matching sequences
            microhomologies.append(left[-(i + 1):])

    return microhomologies

#One Hot encoding including GC frac and MH-count
def dna_to_onehot_extra(dna_sequences, sequence_length=60):
    batch_size = len(dna_sequences)
    num_channels = 6  # A, T, C, G, GC content, Stem Loop Count

    # Initialize a zero tensor of shape (batch_size, 6, sequence_length)
    onehot_matrix = torch.zeros((batch_size, num_channels, sequence_length), dtype=torch.float32, device=device)

    # Define the mapping from nucleotide to channel index
    one_mer_comb_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    for batch_index, dna_sequence in enumerate(dna_sequences):
        # Truncate or pad the sequence to the desired length
        if len(dna_sequence) > sequence_length:
            dna_sequence = dna_sequence[:sequence_length]
        elif len(dna_sequence) < sequence_length:
            dna_sequence = dna_sequence.ljust(sequence_length, 'A')  # Pad with 'A'

        # One-hot encode the sequence
        for i, nucleotide in enumerate(dna_sequence):
            channel_index = one_mer_comb_map.get(nucleotide, None)
            if channel_index is not None:
                onehot_matrix[batch_index, channel_index, i] = 1
            else:
                # Handle invalid nucleotide: currently leaves as zero
                pass

        # Calculate GC content for the sequence
        gc_count = sum(1 for base in dna_sequence if base in 'GC')
        gc_content = gc_count / len(dna_sequence)
        onehot_matrix[batch_index, 4, :] = gc_content  # Add GC content as an additional channel

        # Calculate Microhomology count
        microhomologies = find_microhomologies(dna_sequence)
        mh_count= len(microhomologies)

        onehot_matrix[batch_index, 5, :] = mh_count  # Add microhomology count as another channel

    return onehot_matrix.squeeze(0)


#Simple one Hot encoding
def dna_to_onehot(dna_sequences, sequence_length=60):
    batch_size = len(dna_sequences)
    num_channels = 4  # A, T, C, G

    # Initialize a zero tensor of shape (batch_size, 4, sequence_length)
    onehot_matrix = torch.zeros((batch_size, num_channels, sequence_length), dtype=torch.float32, device=device)

    # Define the mapping from nucleotide to channel index
    one_mer_comb_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    # Iterate over each DNA sequence in the batch
    for batch_index, dna_sequence in enumerate(dna_sequences):
        # Truncate or pad the sequence to the desired length
        if len(dna_sequence) > sequence_length:
            dna_sequence = dna_sequence[:sequence_length]
        elif len(dna_sequence) < sequence_length:
            dna_sequence = dna_sequence.ljust(sequence_length, 'A')  # Padding with 'A' or any nucleotide

        for i, nucleotide in enumerate(dna_sequence):
            channel_index = one_mer_comb_map.get(nucleotide, None)
            if channel_index is not None:
                onehot_matrix[batch_index, channel_index, i] = 1
            else:
                # Handle invalid nucleotide: currently leaves as zero
                pass

    return onehot_matrix.squeeze(0)