def remove_second_column(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split(',')
            if len(parts) == 3:
                # Оставляем только первую и третью колонку
                outfile.write(f"{parts[0]},{parts[2]}\n")

remove_second_column("Test_Near.txt", "Test_Near_Ready.txt")
remove_second_column("Test_Far.txt", "Test_Far_Ready.txt")
