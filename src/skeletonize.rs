use opencv::core::{CV_16U, bitwise_or, count_non_zero, subtract};
use opencv::imgproc::{ContourApproximationModes, RetrievalModes, ThresholdTypes};
use opencv::prelude::*;
use opencv::{
    Result,
    core::{Point, Scalar, Size, Vector},
    highgui, imgcodecs, imgproc,
};

fn skeletonize(img: &Mat) -> Result<Mat> {
    let mut new_img = Mat::default();
    img.copy_to(&mut new_img)?;

    return Ok(new_img);
}
