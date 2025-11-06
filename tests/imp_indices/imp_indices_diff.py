import torch

def compare_imp_indices(file1_path, file2_path):
    """
    Compare the differences in 'imp_indices' between two torch saved files
    
    Parameters:
        file1_path: Path to the first file
        file2_path: Path to the second file
    """
    try:
        # Load the first file
        data1 = torch.load(file1_path)
        if "imp_indices" not in data1:
            raise ValueError(f"Could not find 'imp_indices' key in the first file {file1_path}")
        imp1 = data1["imp_indices"]
        
        # Load the second file
        data2 = torch.load(file2_path)
        if "imp_indices" not in data2:
            raise ValueError(f"Could not find 'imp_indices' key in the second file {file2_path}")
        imp2 = data2["imp_indices"]
        
        # Convert to sets for comparison (assuming elements are hashable)
        # If they are tensors, convert to lists first
        if isinstance(imp1, torch.Tensor):
            imp1 = imp1.tolist()
        if isinstance(imp2, torch.Tensor):
            imp2 = imp2.tolist()
            
        set1 = set(imp1)
        set2 = set(imp2)
        
        # Calculate differences
        file1_extra = set1 - set2  # Elements in first file but not in second
        file1_missing = set2 - set1  # Elements in second file but not in first
        common_elements = set1 & set2  # Elements present in both files
        
        # Output results
        print(f"First file: {file1_path}")
        print(f"Second file: {file2_path}\n")
        
        print(f"Elements in first file but not in second ({len(file1_extra)} items):")
        print(sorted(file1_extra))
        
        print(f"\nElements in second file but not in first ({len(file1_missing)} items):")
        print(sorted(file1_missing))


        print(f"\nElements present in both files ({len(common_elements)} items):")
        print(sorted(common_elements))
            
        # Additional information
        print(f"\nTotal elements in first file: {len(set1)}")
        print(f"Total elements in second file: {len(set2)}")
        print(f"Number of common elements: {len(set1 & set2)}")
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python compare_imp_indices.py <path_to_first_file> <path_to_second_file>")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    compare_imp_indices(file1, file2)