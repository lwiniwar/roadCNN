import numpy as np
import xarray
import rioxarray

import scipy.ndimage as simg
#
# # Chunk Zero (0)
# XMIN = 571484.3997
# XMAX = 609480.9522
# YMIN = 6168623.7861
# YMAX = 6138710.1969

# Chunk Seven (7)
XMIN = 607956
XMAX = 643455
YMIN = 6110475
YMAX = 6081474

def main(ref_file, pred_file, iterations=5):
    out_data = []
    ref_data = rioxarray.open_rasterio(ref_file)
    ref_data = ref_data.sel(x=slice(XMIN, XMAX), y=slice(YMIN, YMAX)).to_numpy()[0].astype(int)
    pred_data = rioxarray.open_rasterio(pred_file) > 0.5
    pred_data = pred_data.sel(x=slice(XMIN, XMAX), y=slice(YMIN, YMAX)).to_numpy()[0].astype(int)

    # LC
    lc_data = rioxarray.open_rasterio(r"C:\Users\Lukas\Documents\Data\road-cnn-small\20230206\tile7_lc_3m_nn.tif")
    lc_data = lc_data.sel(x=slice(XMIN, XMAX), y=slice(YMIN, YMAX)).to_numpy()[0].astype(int)

    # SLOPE
    slope_data = rioxarray.open_rasterio(r"C:\Users\Lukas\Documents\Data\road-cnn-small\20230206\UTM10S_SLOPE_tile7_nn.tif")
    slope_data = slope_data.sel(x=slice(XMIN, XMAX), y=slice(YMIN, YMAX)).to_numpy()[0]

    # ASPECT
    aspect_data = rioxarray.open_rasterio(r"C:\Users\Lukas\Documents\Data\road-cnn-small\20230206\UTM10S_ASPECT_tile7_nn.tif")
    aspect_data = aspect_data.sel(x=slice(XMIN, XMAX), y=slice(YMIN, YMAX)).to_numpy()[0]

    # RGB / NDVI
    rgbi_data = rioxarray.open_rasterio(r"C:\Users\Lukas\Documents\Data\road-cnn-small\aoi_rasters\ps-tile.tif")
    rgbi_data = rgbi_data.sel(x=slice(XMIN, XMAX), y=slice(YMIN, YMAX)).to_numpy().astype(float)
    ndvi = (rgbi_data[3] - rgbi_data[0]) / (rgbi_data[3] + rgbi_data[0])

    # pred_data = np.logical_or(pred_data == 3, pred_data == 1)

    if iterations > 0:
        ref_data_di = simg.binary_dilation(ref_data, iterations=iterations)
        pred_data_di = simg.binary_dilation(pred_data, iterations=iterations)

        # buffer output
        TP_bin = np.logical_and(ref_data_di == 1, pred_data_di == 1)
        TP_buffer = simg.binary_dilation(TP_bin, iterations=2*iterations)
        # exclude actual true positives
        TP_buffer[TP_bin] = 0
        # clean up eval_data
        pred_data[TP_buffer] = 0
        pred_data[TP_bin] = 1
        ref_data[TP_buffer] = 0
        ref_data[TP_bin] = 1

    TPx = np.logical_and(ref_data == 1, pred_data == 1)
    TNx = np.logical_and(ref_data == 0, pred_data == 0)
    FPx = np.logical_and(ref_data == 0, pred_data == 1)
    FNx = np.logical_and(ref_data == 1, pred_data == 0)

    print(f"Comparing {ref_file} with {pred_file}:")
    print(f"Using a dilation of {iterations} px:")

    for NDVImin, NDVImax in zip([-1.0, 0, 0.5, 0.75], [0, 0.5, 0.75, 1]):
        print(f"NDVI from {NDVImin} to {NDVImax}")

        limits = np.logical_and(ndvi >= NDVImin, ndvi < NDVImax)
        TP = np.count_nonzero(np.logical_and(limits, TPx))
        TN = np.count_nonzero(np.logical_and(limits, TNx))
        FP = np.count_nonzero(np.logical_and(limits, FPx))
        FN = np.count_nonzero(np.logical_and(limits, FNx))

        if TP == 0:
            print("\t NA (TP = 0)")
            out_data.append([-1, -1, -1, -1])
            continue
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2* precision * recall / (precision + recall)
        print(f"\tAccuracy  : {accuracy*100:.3f} %")
        print(f"\tPrecision : {precision*100:.3f} %")
        print(f"\tRecall    : {recall*100:.3f} %")
        print(f"\tF1 Score  : {F1*100:.3f} %")
        out_data.append([accuracy*100, precision*100, recall*100, F1*100])

    for slopemin, slopemax in zip([0, 5, 10, 20, 30], [5, 10, 20, 30, 90]):
        print(f"Slope range {slopemin}-{slopemax} deg")

        limits = np.logical_and(slope_data >= slopemin, slope_data < slopemax)
        TP = np.count_nonzero(np.logical_and(limits, TPx))
        TN = np.count_nonzero(np.logical_and(limits, TNx))
        FP = np.count_nonzero(np.logical_and(limits, FPx))
        FN = np.count_nonzero(np.logical_and(limits, FNx))

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2* precision * recall / (precision + recall)
        print(f"\tAccuracy  : {accuracy*100:.3f} %")
        print(f"\tPrecision : {precision*100:.3f} %")
        print(f"\tRecall    : {recall*100:.3f} %")
        print(f"\tF1 Score  : {F1*100:.3f} %")

        out_data.append([accuracy*100, precision*100, recall*100, F1*100])

    for aspectmin, aspectmax in zip([0, 90, 180, 270], [90, 180, 270, 360]):
        print(f"Aspect range  {aspectmin-45} to {aspectmax-45} deg")
        aspect_data = aspect_data + 45
        aspect_data[aspect_data > 360] -= 360

        limits = np.logical_and(aspect_data >= aspectmin, aspect_data < aspectmax)
        TP = np.count_nonzero(np.logical_and(limits, TPx))
        TN = np.count_nonzero(np.logical_and(limits, TNx))
        FP = np.count_nonzero(np.logical_and(limits, FPx))
        FN = np.count_nonzero(np.logical_and(limits, FNx))

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2* precision * recall / (precision + recall)
        print(f"\tAccuracy  : {accuracy*100:.3f} %")
        print(f"\tPrecision : {precision*100:.3f} %")
        print(f"\tRecall    : {recall*100:.3f} %")
        print(f"\tF1 Score  : {F1*100:.3f} %")

        out_data.append([accuracy*100, precision*100, recall*100, F1*100])

    for lc in np.unique(lc_data):
        print(f"LC Class {lc}")

        limits = lc_data == lc
        TP = np.count_nonzero(np.logical_and(limits, TPx))
        TN = np.count_nonzero(np.logical_and(limits, TNx))
        FP = np.count_nonzero(np.logical_and(limits, FPx))
        FN = np.count_nonzero(np.logical_and(limits, FNx))

        if TP > 0:
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1 = 2 * precision * recall / (precision + recall)
            print(f"\tAccuracy  : {accuracy * 100:.3f} %")
            print(f"\tPrecision : {precision * 100:.3f} %")
            print(f"\tRecall    : {recall * 100:.3f} %")
            print(f"\tF1 Score  : {F1 * 100:.3f} %")

            out_data.append([accuracy * 100, precision * 100, recall * 100, F1 * 100])
        else:
            print("\tNo TP (N/A)")

            out_data.append([-1, -1, -1, -1])

    try:
        TP = np.count_nonzero(TPx)
        TN = np.count_nonzero(TNx)
        FP = np.count_nonzero(FPx)
        FN = np.count_nonzero(FNx)
        
        print("Overall scores:")
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2* precision * recall / (precision + recall)
        print(f"\tAccuracy  : {accuracy*100:.3f} %")
        print(f"\tPrecision : {precision*100:.3f} %")
        print(f"\tRecall    : {recall*100:.3f} %")
        print(f"\tF1 Score  : {F1*100:.3f} %")
        return accuracy, precision, recall, F1, out_data
    except Exception as e:
        print("Error:")
        print(e)



if __name__ == '__main__' and False:
    mixers = [
        r"C:\Users\Lukas\Documents\Data\road-cnn-small\20230203\test_pred_gaussian.tif"
        # r"C:\Users\Lukas\Documents\Data\road-cnn-small\test_pred.tif",
        # r"C:\Users\Lukas\Documents\Data\road-cnn-small\test_pred_average.tif",
        # r"C:\Users\Lukas\Documents\Data\road-cnn-small\test_pred_linear.tif",
        # r"C:\Users\Lukas\Documents\Data\road-cnn-small\test_pred_gaussian.tif",
    ]
    for mix in mixers:
        main(r"C:\Users\Lukas\Documents\Data\road-cnn-small\ref_ps_filtered.tif",
             mix, 5)

if __name__ == '__main__':
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    its = [0, 1, 2, 3, 4, 5, 10, 15, 20]
    its = [0, 5]
    out_datas = []
    for iteration in its:
        # main(r"C:\Users\Lukas\Documents\Data\road-cnn-small\ref_re_filtered.tif", r"C:\Users\Lukas\Documents\Data\road-cnn-small\test_pred.tif", iteration)
        accuracy, precision, recall, f1, out_data = main(r"C:\Users\Lukas\Documents\Data\road-cnn-small\ref_ps_manual_t7.tif",
                                               r"C:\Users\Lukas\Documents\Data\road-cnn-small\20230206\tile7_prob_gaussian.tif",
                                               iteration)
        # main(r"C:\Users\Lukas\Documents\Data\road-cnn-small\ref_re_unedited.tif", r"C:\Users\Lukas\Documents\Data\road-cnn-small\test_pred.tif", iteration)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        out_datas.append(np.array(out_data))

    import matplotlib.pyplot as plt

    # plt.plot(its, [acc * 100 for acc in accuracies], "r-o")
    # plt.title("Accuracy over dilation radius")
    # plt.xlabel("Dilation iterations [px]")
    # plt.ylabel("Accuracy [%]")
    # plt.show()
    #
    # plt.plot(its, [acc * 100 for acc in precisions], "g-o")
    # plt.title("Precision over dilation radius")
    # plt.xlabel("Dilation iterations [px]")
    # plt.ylabel("Precision [%]")
    # plt.show()
    #
    # plt.plot(its, [acc * 100 for acc in recalls], "b-o")
    # plt.title("Recall over dilation radius")
    # plt.xlabel("Dilation iterations [px]")
    # plt.ylabel("Recall [%]")
    # plt.show()
    # plt.plot(its, [acc * 100 for acc in f1s], "k-o")
    # plt.title("F1 score over dilation radius")
    # plt.xlabel("Dilation iterations [px]")
    # plt.ylabel("F1 score [%]")
    # plt.show()

    import seaborn as sns
    sns.set()
    sns.set_style("whitegrid")

    plt.figure(figsize=(8, 6))
    plt.plot(its, [acc * 100 for acc in accuracies], '-', label="Accuracy")
    plt.plot(its, [acc * 100 for acc in precisions], '--', label="Precision")
    plt.plot(its, [acc * 100 for acc in recalls], ':', label="Recall")
    plt.plot(its, [acc * 100 for acc in f1s], '-x',  label="F1 score")
    plt.ylabel("Metric score [%]")
    plt.xlabel("Dilation iterations [px]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    for line1, line2 in zip(*out_datas):
        for item1, item2 in zip(line1, line2):
            print(f"{item1:.1f} ({item2:.1f}), ", end='')
        print('')