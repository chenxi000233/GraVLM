import os
import json
import numpy as np
from collections import Counter


# Helper function to extract triples from JSON
def extract_triples(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    triples = []
    for key, value in data.items():
        if key.startswith("triple"):
            triples.append(value)
    return triples


# Helper function to create adjacency matrix and degree matrix
def create_matrices(triples, max_entities=20):
    subject_entities = []  # List to hold subject entities
    object_entities = []  # List to hold object entities
    entity_to_idx = {}  # Mapping from entity to matrix index

    # Extract subjects and objects from triples
    for triple in triples:
        subject, _, object_ = parse_triple(triple)
        subject_entities.append(subject)
        object_entities.append(object_)

    # Combine subject and object entities, remove duplicates, and limit to `max_entities`
    all_entities = list(set(subject_entities + object_entities))[:max_entities]

    # Create a mapping of entities to matrix indices
    for i, entity in enumerate(all_entities):
        entity_to_idx[entity] = i

    # Initialize adjacency matrix as a 20x20 matrix
    adjacency_matrix = np.zeros((max_entities, max_entities), dtype=float)

    # Degree matrix: count occurrences of each entity (node degree)
    degree_matrix = np.zeros((max_entities, max_entities), dtype=float)

    # Build adjacency matrix and degree matrix
    for triple in triples:
        subject, _, object_ = parse_triple(triple)
        if subject in entity_to_idx and object_ in entity_to_idx:
            subject_idx = entity_to_idx[subject]
            object_idx = entity_to_idx[object_]
            adjacency_matrix[subject_idx, object_idx] = 1
            adjacency_matrix[object_idx, subject_idx] = 1  # undirected relation
            degree_matrix[subject_idx, subject_idx] += 1
            degree_matrix[object_idx, object_idx] += 1

    return adjacency_matrix, degree_matrix


# Helper function to parse the triple into subject, predicate, and object
def parse_triple(triple):
    # Assuming the triple is in the format (subject, relation, object)
    subject, relation, object_ = triple.strip('()').split(',')  # Strip parentheses and split by comma
    return subject.strip(), relation.strip(), object_.strip()


# Function to normalize adjacency matrix using D^-1/2 A D^-1/2
def normalize_adjacency_matrix(adj_matrix):
    # Degree matrix D: row sums of adjacency matrix
    row_sum = np.array(adj_matrix.sum(axis=1)).flatten()

    # Add small epsilon to avoid division by zero (ensure no zero degrees)
    epsilon = 1e-8
    row_sum = np.where(row_sum == 0, epsilon, row_sum)  # Replace zeros with epsilon

    # Degree matrix normalization
    degree_matrix = np.diag(1.0 / np.sqrt(row_sum))

    # Standardize adjacency matrix: D^-1/2 A D^-1/2
    adj_matrix_normalized = degree_matrix @ adj_matrix @ degree_matrix

    # Check for NaNs and replace with 0s (optional)
    adj_matrix_normalized = np.nan_to_num(adj_matrix_normalized, nan=0.0)

    return adj_matrix_normalized


# Function to densify adjacency matrix by adding self-loops and small weights
def densify_adjacency_matrix(adj_matrix, epsilon=0.01):
    """
    Densify adjacency matrix by adding self-loops and small weights to sparse regions.
    Args:
        adj_matrix (numpy.ndarray): Original adjacency matrix
        epsilon (float): Small weight value to fill sparse areas
    Returns:
        numpy.ndarray: Densified and normalized adjacency matrix
    """
    adj_matrix = adj_matrix.copy()
    np.fill_diagonal(adj_matrix, 1.0)  # Add self-loops
    adj_matrix[adj_matrix == 0] = epsilon  # Fill sparse areas with a small weight
    return normalize_adjacency_matrix(adj_matrix)


# Main function to process the triples and generate matrices
def process_triples(image_name, triples_dir, max_entities=20):
    json_file_path = os.path.join(triples_dir, f"{image_name}.json")
    if not os.path.exists(json_file_path):
        print(f"Warning: {json_file_path} does not exist.")
        return None, None

    # Extract triples from the JSON file
    triples = extract_triples(json_file_path)
    if not triples:
        print(f"No triples found in {json_file_path}.")
        return None, None

    # Create adjacency matrix and degree matrix
    adjacency_matrix, degree_matrix = create_matrices(triples, max_entities)

    # Densify and normalize adjacency matrix
    adjacency_matrix_densified = densify_adjacency_matrix(adjacency_matrix)

    return adjacency_matrix_densified, degree_matrix


# Example usage
if __name__ == "__main__":
    triples_dir = './triples'  # Path to the directory containing the triples JSON files
    image_name = "airport_504"  # Example image name

    adjacency_matrix, degree_matrix = process_triples(image_name, triples_dir)

    if adjacency_matrix is not None and degree_matrix is not None:
        print("Densified and Normalized Adjacency Matrix (20x20):")
        print(adjacency_matrix)  # Output as a 20x20 matrix
        print("Degree Matrix (20x20):")
        print(degree_matrix)  # Output as a 20x20 matrix
