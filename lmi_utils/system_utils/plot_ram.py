import matplotlib.pyplot as plt
import collections
import itertools

# Function to read and parse the data from the text file
def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() == "" or line.startswith("TOTAL") or line.startswith("Container") or line.startswith("numbers") or line.startswith("Time"):
                continue
            parts = line.split()
            print(parts)
            record = {
                "Container ID": parts[0],
                "Name": parts[1],
                "Total": int(parts[2]),
                "Cache": int(parts[3]),
                "Kernel": int(parts[4]),
                "RSS": int(parts[5]),
                "Swap": int(parts[6])
            }
            data.append(record)
    return data


def plot_and_save(totals, color_marker_combinations, wanted=[], not_wanted=[], outname=None):
    # Plotting the data
    plt.figure(figsize=(10, 6))
    occurrences = []
    for k in totals:
        if any(pattern in k for pattern in not_wanted):
            continue
        if wanted and not any(pattern in k for pattern in wanted):
            continue
        if not occurrences:
            occurrences = list(range(1, len(totals[k]) + 1))
        color,marker = color_marker_combinations.pop(0)
        plt.plot(occurrences, totals[k], marker=marker, color=color, linestyle='-', label=k)
    plt.xlabel('Occurrence')
    plt.ylabel('RSS + Swap (MB)')
    plt.title('RSS + Swap every 30 minutes')
    plt.legend()
    plt.grid(True)
    
    # write plot as png
    if outname:
        plt.savefig(outname)
    plt.close()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True, help="Path to the log file")
    args = ap.parse_args()

    # Read and parse the data
    data = read_data(args.file)

    totals = collections.defaultdict(list)
    for i, record in enumerate(data):
        service_name = record["Name"]
        total_plus_swap = record["RSS"] + record["Swap"]
        totals[service_name].append(total_plus_swap)

    
    # Calculate all possible combinations of colors and markers
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    color_marker_combinations = list(itertools.product(colors, markers))

    plot_and_save(totals, color_marker_combinations, ['pipeline'], outname="ram_pipeline.png")
    plot_and_save(totals, color_marker_combinations, ['sensor'], outname="ram_sensor.png")
    plot_and_save(totals, color_marker_combinations, not_wanted=['pipeline', 'sensor'], outname="ram_gadget.png")
    