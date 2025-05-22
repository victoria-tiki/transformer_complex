import os

image_dir = "waveform_gallery2"
js_array = []

for filename in sorted(os.listdir(image_dir)):
    if filename.lower().endswith((".jpg", ".jpeg")):
        try:
            name_only = os.path.splitext(filename)[0]
            q, s1, s2, theta = map(float, name_only.split("_"))
            js_array.append(
                f'  {{ q: {q}, theta: {theta}, s1: {s1}, s2: {s2}, file: "{filename}" }},'
            )
        except ValueError:
            print(f"⚠️ Skipping malformed filename: {filename}")

print("const waveformData = [")
for line in js_array:
    print(line)
print("];")

