mod skeletonize;

use opencv::core::{bitwise_or, count_non_zero, subtract};
use opencv::imgproc::{ContourApproximationModes, RetrievalModes, ThresholdTypes};
use opencv::prelude::*;
use opencv::{
    Result,
    core::{Point, Scalar, Size, Vector},
    highgui, imgcodecs, imgproc,
};

const KNOT_3_1_V1_PATH: &str = "/Users/djprice/SideProjectsCode/knot-detector/data/knot_3_1_v1.jpg";

// I just realized that this is actually a 5 crossing knot, not 4_1
// Oh well
// I'd rather write this long comment about being wrong than actually fix it
const KNOT_4_1_V1_PATH: &str = "/Users/djprice/SideProjectsCode/knot-detector/data/knot_4_1_v1.jpg";

fn main() -> Result<()> {
    let input_path = KNOT_4_1_V1_PATH;

    let img = imgcodecs::imread(input_path, imgcodecs::IMREAD_COLOR)?;

    // ================================================================================
    // Convert to grayscale
    // ================================================================================
    let mut gray_img = Mat::default();
    imgproc::cvt_color(
        &img,
        &mut gray_img,
        imgproc::COLOR_BGR2GRAY,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // ================================================================================
    // Threshold the grayscale image
    // ================================================================================
    let mut thresholded_img = Mat::default();
    imgproc::threshold(
        &gray_img,
        &mut thresholded_img,
        106.0,
        255.0,
        // (ThresholdTypes::THRESH_OTSU as i32) | (ThresholdTypes::THRESH_BINARY_INV as i32),
        imgproc::THRESH_BINARY_INV,
    )?;

    // ================================================================================
    // Clean up any white specks
    // ================================================================================
    let kernel =
        imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(9, 9), Point::new(-1, -1))?;
    let mut cleaned_img = Mat::default();
    imgproc::morphology_ex(
        &thresholded_img,
        &mut cleaned_img,
        imgproc::MORPH_OPEN, // erode then dilate - this removes small bright specks
        &kernel,
        Point::new(-1, -1), // the center point of the kernel - (-1, -1) just means "center"
        1,
        opencv::core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;

    // ================================================================================
    // Find contours
    // ================================================================================
    let mut contours: Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours(
        &cleaned_img,
        &mut contours,
        RetrievalModes::RETR_EXTERNAL as i32,
        ContourApproximationModes::CHAIN_APPROX_NONE as i32,
        Point::new(0, 0),
    )?;

    // =============================================================================
    // Draw contours on a copy of the original color image
    // =============================================================================
    let mut contour_vis = img.clone();
    imgproc::draw_contours(
        &mut contour_vis,
        &contours,
        -1,                                // draw all contours
        Scalar::new(0.0, 0.0, 255.0, 0.0), // red
        2,                                 // thickness
        imgproc::LINE_8,
        &opencv::core::no_array(),
        0,
        Point::new(0, 0),
    )?;

    // ================================================================================
    // Get polygon from contours
    // ================================================================================
    let mut simplified_all: Vector<Vector<Point>> = Vector::new();
    for c in &contours {
        let eps = 0.00001 * imgproc::arc_length(&c, true)?;
        let mut approx: Vector<Point> = Vector::new();

        imgproc::approx_poly_dp(&c, &mut approx, eps, true)?;

        simplified_all.push(approx);
    }

    // ================================================================================
    // Display polygonal approximation on copy of the original color image
    // ================================================================================
    let mut poly_vis = img.clone();
    imgproc::polylines(
        &mut poly_vis,
        &simplified_all,
        true,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;
    println!("{}", simplified_all.len());

    // ================================================================================
    // Display
    // ================================================================================
    highgui::imshow("original", &img)?;
    highgui::wait_key(0)?;

    highgui::imshow("gray", &gray_img)?;
    highgui::wait_key(0)?;

    highgui::imshow("thresholded", &thresholded_img)?;
    highgui::wait_key(0)?;

    highgui::imshow("cleaned", &cleaned_img)?;
    highgui::wait_key(0)?;

    // ================================================================================
    // Skeletonizing
    // ================================================================================
    let mut distance_transform = Mat::zeros(
        thresholded_img.size()?.height,
        thresholded_img.size()?.width,
        thresholded_img.typ(),
    )?
    .to_mat()?;

    let mut dist = 0;
    while count_non_zero(&thresholded_img)? != 0 {
        println!("{}", dist);

        for r in 1..thresholded_img.rows() - 1 {
            for c in 1..thresholded_img.cols() - 1 {
                let threshold_pixel = thresholded_img.at_2d::<u8>(r, c)?;
                let distance_transform_pixel = distance_transform.at_2d_mut::<u8>(r, c)?;

                if *threshold_pixel == 255 {
                    if *thresholded_img.at_2d::<u8>(r - 1, c - 1)? == 0
                        || *thresholded_img.at_2d::<u8>(r - 1, c)? == 0
                        || *thresholded_img.at_2d::<u8>(r - 1, c + 1)? == 0
                        || *thresholded_img.at_2d::<u8>(r, c - 1)? == 0
                        || *thresholded_img.at_2d::<u8>(r, c + 1)? == 0
                        || *thresholded_img.at_2d::<u8>(r + 1, c - 1)? == 0
                        || *thresholded_img.at_2d::<u8>(r + 1, c)? == 0
                        || *thresholded_img.at_2d::<u8>(r + 1, c + 1)? == 0
                    {
                        *distance_transform_pixel = (dist + 1) * 12; // TODO: remove the + 12
                    }
                }
            }
        }

        for r in 1..thresholded_img.rows() - 1 {
            for c in 1..thresholded_img.cols() - 1 {
                let threshold_pixel = thresholded_img.at_2d_mut::<u8>(r, c)?;
                let distance_transform_pixel = distance_transform.at_2d::<u8>(r, c)?;

                if *distance_transform_pixel > 0 {
                    *threshold_pixel = 0;
                }
            }
        }

        dist += 1;
    }

    highgui::imshow("threshold", &thresholded_img)?;
    highgui::wait_key(0)?;

    highgui::imshow("distance", &distance_transform)?;
    highgui::wait_key(0)?;

    return Ok(());
}
