import pandas as pd
# Task 1 = B1      OR  Parity(B2-B8)
# Task 2 = not(B1) OR  Parity(B2-B8)
# Task 3 = B1      AND Parity(B2-B8)
# Task 4 = not(B1) AND Parity(B2-B8)


# build the dataset
def getParity(n):
    parity = 0
    while n:
        parity = ~parity
        n = n & (n - 1)
    return parity

def get_data():
    tasks = {
            1: [],
            2: [],
            3: [],
            4: []
            }

    mask = (1 << 8) - 1

    for byte in range(256):
        binary = format(byte, '08b')
        little = byte & mask # last 8 bits

        for key, value in tasks.items():
            if key == 1:
                # Task 1
                if (byte & (1 << (8-1))) or (getParity(little) == -1):
                    tasks[1].append(1)
                else:
                    tasks[1].append(0)
            if key == 2:
                # Task 2 
                if not(byte & (1 << (8-1))) or (getParity(little) == -1):
                    tasks[2].append(1)
                else:
                    tasks[2].append(0)
            if key == 3:
                # Task 3
                if (byte & (1 << (8-1))) and (getParity(little) == -1):
                    tasks[3].append(1)
                else:
                    tasks[3].append(0)
            if key == 4:
                # Task 4
                if not(byte & (1 << (8-1))) and (getParity(little) == -1):
                    tasks[4].append(1)
                else:
                    tasks[4].append(0)
    
    return tasks

# another version for testing
# Task 1 = B1      OR  Parity(B4-B8)
# Task 2 = not(B1) OR  Parity(B4-B8)
# Task 3 = B1      AND Parity(B4-B8)
# Task 4 = not(B1) AND Parity(B4-B8)
 
def get_data2():
    tasks = {
            1: [],
            2: [],
            3: [],
            4: []
            }

    mask = (1 << 6) - 1

    for byte in range(256):
        binary = format(byte, '08b')
        little = byte & mask # last 6 bits

        for key, value in tasks.items():
            if key == 1:
                # Task 1
                if (byte & (1 << (6-1))) or (getParity(little) == -1):
                    tasks[1].append(1)
                else:
                    tasks[1].append(0)
            if key == 2:
                # Task 2 
                if not(byte & (1 << (6-1))) or (getParity(little) == -1):
                    tasks[2].append(1)
                else:
                    tasks[2].append(0)
            if key == 3:
                # Task 3
                if (byte & (1 << (6-1))) and (getParity(little) == -1):
                    tasks[3].append(1)
                else:
                    tasks[3].append(0)
            if key == 4:
                # Task 4
                if not(byte & (1 << (6-1))) and (getParity(little) == -1):
                    tasks[4].append(1)
                else:
                    tasks[4].append(0)

    return tasks

data = get_data2()

df = pd.DataFrame.from_dict(data)
df.to_csv('./targets2.csv')




