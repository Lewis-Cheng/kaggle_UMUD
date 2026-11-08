def collect_metadata(pattern):
    """Determine px_per_cm and the bounding box for every image whose path matches pattern."""
    old_a = None
    nt = []
    test_df = []
    prev_dist_list = []
    for path in sorted(glob(f'{competition_dir}/{pattern}')):
        px_per_cm = -1
        l, t, r, b = -1, -1, -1, -1
        id = int(path[-8:-4])
        filetype = path[-3:]
        with Image.open(path) as img:
            a = np.asarray(img)
        if len(a.shape) == 3:
            assert a.shape[2] == 3
        height, width = a.shape[:2]

        # Determine the scale (pixels per cm)
        if filetype == 'png':
            # only in test dataset
            # variable scale (3..6 cm high)
            # images are 2.85 cm wide and have square pixels
            first_tick = np.argmax(a[:, 6].mean(axis=-1) > 50) # 117
            second_minor_tick = 150 + np.argmax(a[150:, 6].mean(axis=-1) > 50)
            second_major_tick = 150 + np.argmax(a[150:, 9].mean(axis=-1) > 50)
            last_tick = len(a) - 1 - np.argmax(a[:, 6].mean(axis=-1)[::-1] > 50)
            if (second_major_tick - first_tick) < 3 * (second_minor_tick - first_tick):
                px_per_cm = second_major_tick - first_tick # major ticks are 10 mm, minor ticks are 5 mm
            else:
                px_per_cm = second_minor_tick - first_tick # major ticks are 50 mm, minor ticks are 10 mm
            hw = a.shape[1] // 2
            w2 = np.argmin(a[:, hw:].sum(axis=(0, 2)))
            l, t, r, b = hw - w2, first_tick, hw + w2, last_tick
            assert 2.8 < (r - l) / px_per_cm < 2.9 # image width in cm
        elif a.shape == (800, 1200, 3):
            # tick marks can be left, right, and at the bottom; train images can be hflipped; pixels can be non-square
            if (a[87, 1147:1157] == 175).all():
                # variable scale (3..7 cm high), tick marks at right border
                # images in test dataset are 5.75 cm wide and have square pixels
                assert (a[87, 1147:1157] == 175).all() # first tick
                first_tick = np.argmax(a[:, 1150].mean(axis=-1) > 50) # 87
                # assert first_tick == 87, first_tick
                second_major_tick = first_tick + 20 + np.argmax(a[first_tick + 20:, 1150].mean(axis=-1) > 50)
                last_tick = len(a) - 1 - np.argmax(a[:, 1150].mean(axis=-1)[::-1] > 50)
                n_ticks = (last_tick - first_tick) / (second_major_tick - first_tick)
                nt.append(n_ticks)
                n_ticks = round(n_ticks)
                if n_ticks <= 14:
                    px_per_cm = (last_tick - first_tick) / n_ticks * 2 # <= 14 ticks -> 2 ticks per cm
                    if n_ticks == 7:
                        l, t, r, b = 142, 91, 1058, last_tick # 3.5 cm
                    elif n_ticks == 8:
                        l, t, r, b = 163, 91, 1037, last_tick # 4 cm
                    elif n_ticks == 9:
                        l, t, r, b = 211, 91, 989, last_tick # 4.5 cm
                    elif n_ticks == 10:
                        l, t, r, b = 249, 91, 951, last_tick # 5 cm
                    elif n_ticks == 11:
                        l, t, r, b = 282, 91, 918, last_tick # 5.5 cm
                    elif n_ticks == 12:
                        l, t, r, b = 308, 91, 892, last_tick # 6 cm
                    elif n_ticks == 13:
                        l, t, r, b = 331, 91, 869, last_tick # 6.5 cm
                    elif n_ticks == 14:
                        l, t, r, b = 349, 91, 851, last_tick # 7 cm
                    else:
                        assert False
                    assert 5.7 < (r - l) / px_per_cm < 5.8 # image width in cm
                elif n_ticks == 15:
                    px_per_cm = (last_tick - first_tick) / 3 # 15 ticks = 3 cm -> 5 ticks per cm
                    l, t, r, b = 142, 91, 1058, last_tick # 3 cm
                    assert 5.7 < (r - l) / px_per_cm < 5.8 # image width in cm
                else:
                    assert False
            elif  (a[42, 67:74, 0] > 115).all():
                # 42px = 0 cm, 783px = 5 cm, tick marks at left border
                px_per_cm = (783 - 42) / 5
                l, t, r, b = 171, 42, 1029, 798 # ticks left 5 cm
            else:
                assert False
        elif a.shape == (644, 1088, 3):
            # 0px = 0cm, 630.5px = 5 cm
            px_per_cm = 630.5 / 5
            l, t, r, b = 140, 0, 947, 643 # ticks left 5 cm
        elif a.shape[0] in [512, 513]:
            # fixed scale
            if a[-5, 49, 0] == 168 and a[-5, 438, 0] == 168:
                px_per_cm = (442 - 53) / 5
            elif a[-5, 52, 0] == 168 and a[-5, 441, 0] == 168:
                px_per_cm = (442 - 53) / 5
            elif a[-5, 53, 0] == 168 and a[-5, 442, 0] == 168:
                px_per_cm = (442 - 53) / 5
            else:
                assert False
            l, t, r, b = 0, 0, width, height - 10
        elif a.shape[0] == 853:
            assert len(a.shape) == 2
            if a[-5, 100] == 170 and a[-5, 934] == 170:
                px_per_cm = (934 - 100) / 5
            elif a[-5, 44] == 170 and a[-5, 879] == 170:
                px_per_cm = (879 - 44) / 5
            else:
                assert False
            l, t, r, b = 0, 0, width, height
        else:
            assert False, "Unknown format"
            
        test_df.append((id, 
                        path.split('/')[-1],
                        filetype,
                        a.shape[0], a.shape[1],
                        len(a.shape),
                        px_per_cm,
                        l, t, r, b))
        # print(f"{id:4}.{filetype}: {a.shape[0]:5} {a.shape[1]:5} {len(a.shape)} {px_per_cm:7.2f}")
    
    test_df = pd.DataFrame(test_df, columns=['id', 'image_id', 'filetype', 'height', 'width', 'dim', 'px_per_cm', 'l', 't', 'r', 'b'])
    print(f'good: {((test_df.px_per_cm > 0) & (test_df.l >= 0)).mean():.0%}')
    return test_df
