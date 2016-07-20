/// HEADER
#include "pcl_hog_extractor.hpp"

/// PROJECT
#include <csapex/msg/io.h>
#include <csapex/utility/register_apex_plugin.h>
#include <csapex/param/parameter_factory.h>
#include <csapex/model/node_modifier.h>
#include <csapex/msg/generic_vector_message.hpp>
#include <csapex_opencv/roi_message.h>
#include <csapex_opencv/cv_mat_message.h>
#include <csapex_ml/features_message.h>

#include <pcl/point_types.h>
#include "../../include/person_detection_pcl/person_classifier.h"

CSAPEX_REGISTER_CLASS(person_detection::PCLHOGExtractor, csapex::Node)

using namespace csapex;
using namespace csapex::connection_types;
using namespace person_detection;

namespace
{
    struct HardcodedDefaults
    {
        static const int WINDOW_WIDTH       = 64;
        static const int WINDOW_HEIGHT      = 128;
        static const int DESCRIPTOR_SIZE    = 3024;
    };
}

PCLHOGExtractor::PCLHOGExtractor()
{
}

void PCLHOGExtractor::setupParameters(Parameterizable& parameters)
{
    /*
    parameters.addParameter(param::ParameterFactory::declareRange("hog/window_width",
                                                                  param::ParameterDescription("Window width."),
                                                                  1, 512, 64, 1),
                            window_width_);
    parameters.addParameter(param::ParameterFactory::declareRange("hog/window_height",
                                                                  param::ParameterDescription("Window height."),
                                                                  1, 512, 128, 1),
                            window_height_);
    parameters.addParameter(param::ParameterFactory::declareRange("hog/bin_size",
                                                                  param::ParameterDescription("Spatial bin size."),
                                                                  4, 16, 8, 1),
                            bin_size_);
    parameters.addParameter(param::ParameterFactory::declareRange("hog/n_orients",
                                                                  param::ParameterDescription("Number of orientation bins."),
                                                                  1, 18, 9, 1),
                            n_orients_);
    parameters.addParameter(param::ParameterFactory::declareBool("hog/soft_bin",
                                                                 param::ParameterDescription("If true, each pixel can contribute to multiple spatial bins (using bilinear interpolation)."),
                                                                 true),
                            soft_bin_);
    */

    parameters.addParameter(param::ParameterFactory::declareBool("mirror", true),
                            mirror_);

}

void PCLHOGExtractor::setup(NodeModifier& node_modifier)
{
    in_img_     = node_modifier.addInput<CvMatMessage>("image");
    in_rois_    = node_modifier.addInput<GenericVectorMessage, RoiMessage>("rois");
    out_        = node_modifier.addOutput<GenericVectorMessage, FeaturesMessage>("features");
}

void PCLHOGExtractor::process()
{
    CvMatMessage::ConstPtr  in = msg::getMessage<CvMatMessage>(in_img_);
    std::shared_ptr<std::vector<RoiMessage> const> in_rois =
            msg::getMessage<GenericVectorMessage, RoiMessage>(in_rois_);
    std::shared_ptr<std::vector<FeaturesMessage>> out(new std::vector<FeaturesMessage>);

    if (in->value.channels() != 3) {
        throw std::runtime_error("Only 3 channel matrices supported!");
    }

    using ResizerType = pcl_backport::people::PersonClassifier<pcl::RGB>;
    using Cloud = ResizerType::PointCloud;
    using CloudPtr = ResizerType::PointCloudPtr;

    auto extract_feature = [this](CloudPtr& image, const Roi& roi)
    {
        ResizerType resizer;

        float height_person = roi.h();
        float xc = roi.x() + roi.w() / 2;
        float yc = roi.y() + roi.h() / 2;

        int height = floor((height_person * HardcodedDefaults::WINDOW_HEIGHT) / (0.75 * HardcodedDefaults::WINDOW_HEIGHT) + 0.5);  // floor(i+0.5) = round(i)
        int width = floor((height_person * HardcodedDefaults::WINDOW_WIDTH) / (0.75 * HardcodedDefaults::WINDOW_HEIGHT) + 0.5);
        int xmin = floor(xc - width / 2 + 0.5);
        int ymin = floor(yc - height / 2 + 0.5);

        CloudPtr box(new Cloud());
        resizer.copyMakeBorder(image, box, xmin, ymin, width, height);

        CloudPtr sample(new Cloud());
        resizer.resize(box, sample, HardcodedDefaults::WINDOW_WIDTH, HardcodedDefaults::WINDOW_HEIGHT);

        std::unique_ptr<float[]> sample_float(new float[sample->width * sample->height * 3]);
        int delta = sample->height * sample->width;
        for (uint32_t row = 0; row < sample->height; row++)
        {
            for (uint32_t col = 0; col < sample->width; col++)
            {
                sample_float[row + sample->height * col] = ((float) ((*sample)(col, row).r))/255; //ptr[col * 3 + 2];
                sample_float[row + sample->height * col + delta] = ((float) ((*sample)(col, row).g))/255; //ptr[col * 3 + 1];
                sample_float[row + sample->height * col + delta * 2] = (float) (((*sample)(col, row).b))/255; //ptr[col * 3];
            }
        }

        FeaturesMessage feature;
        feature.classification = roi.classification();
        feature.value.resize(HardcodedDefaults::DESCRIPTOR_SIZE);
        hog_.compute(sample_float.get(), feature.value.data());

        return feature;
    };

    const cv::Mat& image = in->value;
    CloudPtr pcl_image(new Cloud(image.cols, image.rows));
    for (std::size_t h = 0; h < pcl_image->height; ++h)
        for (std::size_t w = 0; w < pcl_image->width; ++w)
        {
            pcl::RGB rgb;
            // assume bgr
            rgb.r = image.at<cv::Vec3b>(cv::Point(w, h))[2];
            rgb.g = image.at<cv::Vec3b>(cv::Point(w, h))[1];
            rgb.b = image.at<cv::Vec3b>(cv::Point(w, h))[0];

            pcl_image->at(w, h) = rgb;
        }

    CloudPtr pcl_mirrored;
    if (mirror_)
    {
        cv::Mat mirrored;
        cv::flip(image, mirrored, 1);

        pcl_mirrored = CloudPtr(new Cloud(mirrored.cols, mirrored.rows));
        for (std::size_t h = 0; h < pcl_mirrored->height; ++h)
            for (std::size_t w = 0; w < pcl_mirrored->width; ++w)
            {
                pcl::RGB rgb;
                // assume bgr
                rgb.r = mirrored.at<cv::Vec3b>(cv::Point(w, h))[2];
                rgb.g = mirrored.at<cv::Vec3b>(cv::Point(w, h))[1];
                rgb.b = mirrored.at<cv::Vec3b>(cv::Point(w, h))[0];

                pcl_mirrored->at(w, h) = rgb;
            }
    }

    for (auto& roi_msg : *in_rois)
    {
        Roi roi = roi_msg.value;
        roi.setRect(roi.rect() & cv::Rect(0, 0, image.cols, image.rows));

        out->push_back(extract_feature(pcl_image, roi));

        if (mirror_)
        {
            cv::Rect rect = roi.rect();
            rect.x = image.cols - rect.x - rect.width;
            roi.setRect(rect);
            out->push_back(extract_feature(pcl_mirrored, roi));
        }
    }

    msg::publish<GenericVectorMessage, FeaturesMessage>(out_, out);
}
