import h5py

def merge_h5_files(file1_path, file2_path, output_path):
    """
    Merge two h5py files with structure:
        {aep_model}/{wake_model}/start_{start_id}
    file1 contains start_id 0-9, file2 contains start_id 10-19.
    Writes the combined result to output_path.
    """
    with h5py.File(output_path, "w") as fout:
        # Copy everything from file1 as-is
        with h5py.File(file1_path, "r") as f1:
            for key in f1.keys():
                f1.copy(key, fout)

        # Copy everything from file2, merging into existing groups where needed
        with h5py.File(file2_path, "r") as f2:
            def copy_recursive(src_group, dst_group):
                for name, item in src_group.items():
                    if isinstance(item, h5py.Group):
                        if name in dst_group:
                            # Group already exists (e.g. aep_model or wake_model level)
                            # -> recurse into it instead of overwriting
                            copy_recursive(item, dst_group[name])
                        else:
                            src_group.copy(name, dst_group)
                    else:
                        # Dataset - only copy if it doesn't already exist
                        if name not in dst_group:
                            src_group.copy(name, dst_group)

            copy_recursive(f2, fout)

    print(f"Merged file written to: {output_path}")


if __name__ == "__main__":
    merge_h5_files("results/optimization/windfarm_Hornsea_Project_3_HOW03.h5", "results/optimization/Iteration_3_4WF_20starts/windfarm_Hornsea_Project_3_HOW03.h5", "windfarm_Hornsea_Project_3_HOW03.h5")