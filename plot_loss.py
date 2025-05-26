import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

def extract_data(file_path):
    """Extracts loss values, steps, total time and memory from a JSON file."""
    loss_values = []
    steps = []
    total_time = 0
    total_steps = 0
    total_memory = 0
    map_values = []

    with open(file_path, "r") as file:
        for line in file:
            try:
                # Parse each JSON object
                data = json.loads(line)
                # Check if "loss" and "step" exist in the JSON object
                if "loss" in data and "iter" in data:
                    loss_values.append(data["loss"])
                    steps.append(data["iter"])
                # Check if "time" and "memory" exist in the JSON object    
                if "time" in data and "memory" in data:
                    total_steps += 1
                    total_time += data["time"]
                    total_memory += data["memory"]
                if "coco/bbox_mAP" in data:
                    map_values.append(data["coco/bbox_mAP"])


            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line.strip()}")

    return steps, loss_values, total_time/total_steps, total_memory/total_steps, map_values

# File paths for the JSON files of scalars
file1 = 'D:/Repositories/test/25-05-09/yolov3/20250510_185437/vis_data/scalars.json'
file2 = 'D:/Repositories/test/25-05-09/fasterrcnn/20250510_155854/vis_data/scalars.json'
file3 = 'D:/Repositories/test/25-05-09/rtmdet/20250511_013403/vis_data/scalars.json'
file4 = 'D:/Repositories/test/25-05-09/efficientdet/20250510_214430/vis_data/scalars.json'
file5 = 'D:/Repositories/test/25-05-09/ssd300/20250510_193358/vis_data/scalars.json'

# Extract data from files
steps1, loss_values1, total_time1, total_memory1, map_values1 = extract_data(file1)
steps2, loss_values2, total_time2, total_memory2, map_values2 = extract_data(file2)
steps3, loss_values3, total_time3, total_memory3, map_values3 = extract_data(file3)
steps4, loss_values4, total_time4, total_memory4, map_values4 = extract_data(file4)
steps5, loss_values5, total_time5, total_memory5, map_values5 = extract_data(file5)

# Print mean train times and memory for files
print(f"Totals File 1: {total_time1:.2f} {total_memory1:.2f}")
print(f"Totals File 2: {total_time2:.2f} {total_memory2:.2f}")
print(f"Totals File 2: {total_time3:.2f} {total_memory3:.2f}")
print(f"Totals File 2: {total_time4:.2f} {total_memory4:.2f}")
print(f"Totals File 2: {total_time5:.2f} {total_memory5:.2f}")

# Plot comparison of loss or mAP values
plt.figure(figsize=(10, 6))

# plt.plot(steps1, loss_values1, linestyle="-", label="YOLOv3")
# plt.plot(steps2, loss_values2, linestyle="-", label="Faster R-CNN")
# plt.plot(steps3, loss_values3, linestyle="-", label="RTMDet-m")
# plt.plot(steps4, loss_values4, linestyle="-", label="EfficientDet-d3")
# plt.plot(steps5, loss_values5, linestyle="-", label="SSD300")

plt.plot(map_values1, linestyle="-", label="YOLOv3")
plt.plot(map_values2, linestyle="-", label="Faster R-CNN")
plt.plot(map_values3, linestyle="-", label="RTMDet-m")
plt.plot(map_values4, linestyle="-", label="EfficientDet-d3")
plt.plot(map_values5, linestyle="-", label="SSD300")

plt.title("Nuo nulio be balansavimo be duomenų papildymo")
plt.xlabel("Žingsnis")
plt.ylabel("Nuostolis")
plt.legend()
plt.grid(True)
plt.show()
