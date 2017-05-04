#include "person_detection_pcl.hpp"

#include <csapex/msg/io.h>
#include <csapex/msg/input.h>
#include <csapex/msg/output.h>
#include <csapex/model/node_modifier.h>
#include <csapex/param/parameter_factory.h>
#include <csapex/utility/register_apex_plugin.h>
#include <csapex/profiling/timer.h>
#include <csapex/profiling/interlude.hpp>

#include <csapex_point_cloud/msg/indeces_message.h>
#include <csapex/msg/generic_vector_message.hpp>
#include <csapex_transform/transform_message.h>
#include <csapex_opencv/cv_mat_message.h>
#include <csapex_opencv/roi_message.h>

#include <tf_conversions/tf_eigen.h>

CSAPEX_REGISTER_CLASS(person_detection::PCLPersonDetector, csapex::Node)

using namespace csapex;
using namespace csapex::connection_types;
using namespace person_detection;

PCLPersonDetector::PCLPersonDetector()
{
    ground_plane_.resize(4);
    ground_plane_ << 0.0, 0, 1.0, 0.0;
}

void PCLPersonDetector::setupParameters(Parameterizable& parameters)
{
    /// Preprocessing options
    addParameter(param::ParameterFactory::declareRange("preprocess/sampling_factor",
                                                       param::ParameterDescription("Value of the downsampling factor (in each dimension) which is applied to the raw point cloud (default = 1.)"),
                                                       1, 64, 1, 1),
                 std::bind(&PCLPersonDetector::update_detector, this));
    addParameter(param::ParameterFactory::declareRange("preprocess/voxel_size",
                                                       param::ParameterDescription("Value of the voxel dimension (default = 0.06m.)"),
                                                       0.0, 2.0, 0.06, 0.01),
                 std::bind(&PCLPersonDetector::update_detector, this));
    addParameter(param::ParameterFactory::declareInterval("preprocess/fov_limit",
                                                          param::ParameterDescription("Set the field of view of the point cloud in z direction."),
                                                          0.0, 100.0, 0.0, 100.0, 0.01),
                 std::bind(&PCLPersonDetector::update_detector, this));

    /// unused option in code...
//    addParameter(param::ParameterFactory::declareBool("head_centroid",
//                                                      param::ParameterDescription("Set the location of the person centroid (head or body center) (default = true)"),
//                                                      true),
//                 std::bind(&PCLPersonDetector::update_detector, this));
    /// clustering options
    addParameter(param::ParameterFactory::declareInterval("cluster/height_limit",
                                                          param::ParameterDescription("Minimum/Maximum allowed height for a person cluster (default = 1.3/2.3)"),
                                                          0.0, 3.0, 1.3, 2.3, 0.01),
                 std::bind(&PCLPersonDetector::update_detector, this));
    addParameter(param::ParameterFactory::declareInterval("cluster/width_limit",
                                                          param::ParameterDescription("Minimum/Maximum width for a person cluster (default = 0.1/8.0)"),
                                                          0.0, 10.0, 0.1, 8.0, 0.01),
                 std::bind(&PCLPersonDetector::update_detector, this));
    addParameter(param::ParameterFactory::declareRange("cluster/minimum_head_distance",
                                                       param::ParameterDescription("Minimum allowed distance between persons' heads (default = 0.3)"),
                                                       0.0, 2.0, 0.30, 0.01),
                 std::bind(&PCLPersonDetector::update_detector, this));


    /// classifier
    addParameter(param::ParameterFactory::declareBool("use_external_classifier",
                                                      param::ParameterDescription("Use external Cluster classifier"),
                                                      true),
                 std::bind(&PCLPersonDetector::update_detector, this));
    addConditionalParameter(param::ParameterFactory::declareFileInputPath("svm/path",
                                                                          param::ParameterDescription("Trained HOG person classifier svm"),
                                                                          ""),
                            [this]() { return !readParameter<bool>("use_external_classifier"); },
                            std::bind(&PCLPersonDetector::update_detector, this));
    addConditionalParameter(param::ParameterFactory::declareRange("svm/threshold",
                                                                  param::ParameterDescription("SVM value threshold"),
                                                                  -1000.0, 1000.0, 0.0, 0.01),
                            [this]() { return !readParameter<bool>("use_external_classifier"); },
                            param.svm_threshold);

    /// Intrinsic camera calibration
    addParameter(param::ParameterFactory::declareValue("camera/fov_x",
                                                       param::ParameterDescription("RGB camera fov x-axis instrinsic"),
                                                       60.0),
                 std::bind(&PCLPersonDetector::update_detector, this));
    addParameter(param::ParameterFactory::declareValue("camera/fov_y",
                                                       param::ParameterDescription("RGB camera fov y-axis instrinsic"),
                                                       60.0),
                 std::bind(&PCLPersonDetector::update_detector, this));
    addParameter(param::ParameterFactory::declareValue("camera/c_x",
                                                       param::ParameterDescription("RGB camera sensor center x-axis instrinsic"),
                                                       0),
                 std::bind(&PCLPersonDetector::update_detector, this));
    addParameter(param::ParameterFactory::declareValue("camera/c_y",
                                                       param::ParameterDescription("RGB camera sensor center y-axis instrinsic"),
                                                       0),
                 std::bind(&PCLPersonDetector::update_detector, this));
    addParameter(param::ParameterFactory::declareBool("camera/sensor_potrait_orientation",
                                                      param::ParameterDescription("Set landscape/portait camera orientation (default = false)"),
                                                      false),
                 std::bind(&PCLPersonDetector::update_detector, this));
}

void PCLPersonDetector::setup(NodeModifier& node_modifier)
{
    in_cloud_     = node_modifier.addInput<PointCloudMessage>("PointCloud");
    in_ground_    = node_modifier.addOptionalInput<GenericVectorMessage, double>("Ground Plane Coeff");
    in_transfrom_ = node_modifier.addOptionalInput<TransformMessage>("Transform");

    out_clusters_           = node_modifier.addOutput<GenericVectorMessage, pcl::PointIndices>("Clusters");
    out_clusters_euclidean_ = node_modifier.addOutput<GenericVectorMessage, pcl::PointIndices>("Euclidean Clusters");
    out_no_ground_cloud_    = node_modifier.addOutput<PointCloudMessage>("No Ground Cloud");
    out_ground_cloud_       = node_modifier.addOutput<PointCloudMessage>("Ground Cloud");
    out_ground_             = node_modifier.addOutput<GenericVectorMessage, double>("Ground Plane Coeff");

    out_image_                 = node_modifier.addOutput<CvMatMessage>("Image");
    out_clusters_unclassified_ = node_modifier.addOutput<GenericVectorMessage, pcl::PointIndices>("Unclassified Clusters");
    out_rois_unclassified_     = node_modifier.addOutput<GenericVectorMessage, RoiMessage>("Unclassified ROIs");
}

void PCLPersonDetector::process()
{
    PointCloudMessage::ConstPtr pcl_msg(msg::getMessage<PointCloudMessage>(in_cloud_));
    tmp.frame = pcl_msg->frame_id;
    tmp.stamp = pcl_msg->stamp_micro_seconds;
    boost::apply_visitor(PointCloudMessage::Dispatch<PCLPersonDetector>(this, pcl_msg), pcl_msg->value);
}

void PCLPersonDetector::update_classifier()
{
    std::string path = readParameter<std::string>("svm/path");
    if (path.empty())
        return;

    classifier_ = classifier_t();
    classifier_.loadSVMFromFile(path);
}

void PCLPersonDetector::update_intrinsics()
{
    auto fov_x = readParameter<double>("camera/fov_x");
    auto fov_y = readParameter<double>("camera/fov_y");
    auto c_x = readParameter<int>("camera/c_x");
    auto c_y = readParameter<int>("camera/c_y");

    Eigen::Matrix3f instrinsic;
    instrinsic << fov_x, 0.0, c_x,
                  0.0, fov_y, c_y,
                  0.0, 0.0, 1.0;

    instrinsics_ = instrinsic;
}

void PCLPersonDetector::update_detector()
{
    update_classifier();
    update_intrinsics();

    // reset
    detector_ = detector_t();
    // update preprocessing
    detector_.setSamplingFactor(readParameter<int>("preprocess/sampling_factor"));
    detector_.setVoxelSize(readParameter<double>("preprocess/voxel_size"));
    auto fov_limits = readParameter<std::pair<double, double>>("preprocess/fov_limit");
    detector_.setFOV(fov_limits.first, fov_limits.second);
    // update clustering
    auto height_limits = readParameter<std::pair<double, double>>("cluster/height_limit");
    auto width_limits = readParameter<std::pair<double, double>>("cluster/width_limit");
    detector_.setPersonClusterLimits(height_limits.first, height_limits.second, width_limits.first, width_limits.second);
    detector_.setMinimumDistanceBetweenHeads(readParameter<double>("cluster/minimum_head_distance"));
    // update camera
    detector_.setIntrinsics(instrinsics_);
    detector_.setSensorPortraitOrientation(readParameter<bool>("camera/sensor_potrait_orientation"));
    // update classifier
    if (!readParameter<bool>("use_external_classifier"))
        detector_.setClassifier(classifier_);
    // unused
//    detector_.setHeadCentroid(readParameter<bool>("head_centroid"));
}

template<typename PointT>
void PCLPersonDetector::inputCloud(typename pcl::PointCloud<PointT>::ConstPtr cloud)
{
    detail::PCLDispatch<PointT> dispatcher;
    dispatcher(this, cloud);
}

void PCLPersonDetector::process(typename pcl::PointCloud<::pcl::PointXYZRGBA>::ConstPtr cloud)
{
    cloud_t::Ptr copy = boost::make_shared<cloud_t>();
    {
        INTERLUDE("conversion");
        pcl::copyPointCloud(*cloud, *copy);
    }

    this->process(copy);
}

void PCLPersonDetector::process(typename pcl::PointCloud<::pcl::PointXYZRGBL>::ConstPtr cloud)
{
    cloud_t::Ptr copy = boost::make_shared<cloud_t>();
    {
        INTERLUDE("conversion");
        pcl::copyPointCloud(*cloud, *copy);
    }

    this->process(copy);
}

void PCLPersonDetector::process(typename pcl::PointCloud<::pcl::PointXYZI>::ConstPtr cloud)
{
    cloud_t::Ptr copy = boost::make_shared<cloud_t>(cloud->width, cloud->height);
    {
        INTERLUDE("conversion");
        for (uint32_t y = 0; y < copy->height; ++y)
            for (uint32_t x = 0; x < copy->width; ++x)
            {
                const auto& src = cloud->at(x, y);
                point_t pt;
                if (std::isfinite(src.x) && std::isfinite(src.y) && std::isfinite(src.z))
                {
                    pt.x = src.x;
                    pt.y = src.y;
                    pt.z = src.z;
                }
                else
                {
                    pt.x = 0;
                    pt.y = 0;
                    pt.z = 0;
                }
                pt.r = src.intensity;
                pt.g = src.intensity;
                pt.b = src.intensity;
                copy->at(x, y) = pt;
            }
    }

    this->process(copy);
}

void PCLPersonDetector::process(typename pcl::PointCloud<::pcl::PointXYZRGB>::ConstPtr cloud)
{
    cloud_t::Ptr uneccessary_copy_because_of_stupid_api;
    {
        INTERLUDE("conversion");
        uneccessary_copy_because_of_stupid_api = boost::make_shared<cloud_t>(*cloud);
    }

    this->process(uneccessary_copy_because_of_stupid_api);
}

void PCLPersonDetector::process(cloud_t::Ptr cloud)
{
    std::vector<cluster_t> clusters;
    {
        INTERLUDE("detection");
        // update ground if available
        if (msg::hasMessage(in_ground_))
        {
            std::shared_ptr<const std::vector<double>> ground_msg = msg::getMessage<GenericVectorMessage, double>(in_ground_);
            if (ground_msg->size() != 4)
                throw std::runtime_error("Ground Plane must have 4 coeffs (n_x, n_y, n_z, d)");

            for (int i = 0; i < 4; ++i)
                ground_plane_[i] = (*ground_msg)[i];
        }


        detector_.setInputCloud(cloud);
        detector_.setGround(ground_plane_);

        // update transformation if available
        if (msg::isConnected(in_transfrom_))
        {
            TransformMessage::ConstPtr tf_msg = msg::getMessage<TransformMessage>(in_transfrom_);
            const tf::Transform& tf = tf_msg->value;
            Eigen::Matrix3d rot;
            tf::matrixTFToEigen(tf.getBasis(), rot);
            detector_.setTransformation(rot.cast<float>());
        }
        else
        {
            detector_.reset_transformation();
        }

        detector_.compute(clusters);
        ground_plane_ = detector_.getGround();
    }

    // output (final) clusters
    if (msg::isConnected(out_clusters_))
    {
        INTERLUDE("output_clusters");
        std::shared_ptr<std::vector<pcl::PointIndices>> msg_clusters = std::make_shared<std::vector<pcl::PointIndices>>();
        const double confidence_threshold = readParameter<double>("svm/threshold");
        for (cluster_t& cluster : clusters)
        {
            if (cluster.getPersonConfidence() < confidence_threshold)
                continue;

            msg_clusters->push_back(cluster.getIndices());
        }
        msg::publish<GenericVectorMessage, pcl::PointIndices>(out_clusters_, msg_clusters);
    }

    // output (euclidean) clusters
    if (msg::isConnected(out_clusters_euclidean_))
    {
        INTERLUDE("output_euclidean_clusters");
        std::shared_ptr<std::vector<pcl::PointIndices>> msg_clusters_pre = std::make_shared<std::vector<pcl::PointIndices>>();
        *msg_clusters_pre = detector_.get_clustering_indieces();
        msg::publish<GenericVectorMessage, pcl::PointIndices>(out_clusters_euclidean_, msg_clusters_pre);
    }

    // output ground plane coeff
    if (msg::isConnected(out_ground_))
    {
        INTERLUDE("output_ground_plane");
        std::shared_ptr<std::vector<double>> ground_msg = std::make_shared<std::vector<double>>();
        for (int i = 0; i < 4; ++i)
            ground_msg->push_back(ground_plane_[i]);
        msg::publish<GenericVectorMessage, double>(out_ground_, ground_msg);
    }

    // output no ground cloud
    if (msg::isConnected(out_no_ground_cloud_))
    {
        INTERLUDE("output_no_ground_cloud");
        PointCloudMessage::Ptr no_ground_cloud = std::make_shared<PointCloudMessage>(tmp.frame, tmp.stamp);
        no_ground_cloud->value = detector_.getNoGroundCloud();
        msg::publish(out_no_ground_cloud_, no_ground_cloud);
    }

    // output ground cloud
    if (msg::isConnected(out_ground_cloud_))
    {
        INTERLUDE("output_ground_cloud");
        PointCloudMessage::Ptr ground_cloud = std::make_shared<PointCloudMessage>(tmp.frame, tmp.stamp);
        ground_cloud->value = detector_.getGroundCloud();
        msg::publish(out_ground_cloud_, ground_cloud);
    }

    // output unclassified
    if (msg::isConnected(out_clusters_unclassified_) || msg::isConnected(out_rois_unclassified_) || msg::isConnected(out_image_))
    {
        INTERLUDE("output_unclassified");
        std::shared_ptr<std::vector<pcl::PointIndices>> msg_clusters = std::make_shared<std::vector<pcl::PointIndices>>();
        std::shared_ptr<std::vector<RoiMessage>> msg_rois = std::make_shared<std::vector<RoiMessage>>();
        for (cluster_t& cluster : clusters)
        {

            RoiMessage roi;
            if (!readParameter<bool>("camera/sensor_potrait_orientation"))
            {
                float pixel_height;
//                float pixel_width;

                pixel_height = cluster.projected_bottom_(1) - cluster.projected_top_(1);
//                pixel_width = pixel_height / 2.0f;

                float pixel_xc = cluster.projected_center_(0);
                float pixel_yc = cluster.projected_center_(1);

                int height = pixel_height;
                int width = pixel_height / 2;
                int xmin = floor(pixel_xc - width / 2 + 0.5);
                int ymin = floor(pixel_yc - height / 2 + 0.5);

                roi.value.setX(xmin);
                roi.value.setY(ymin);
                roi.value.setW(width);
                roi.value.setH(height);
            }
            else
            {
//                float pixel_height;
                float pixel_width;

                pixel_width = cluster.projected_top_(0) - cluster.projected_bottom_(1);
//                pixel_height = pixel_width / 2.0f;

                float p_pixel_xc = cluster.projected_center_(0);
                float p_pixel_yc = cluster.projected_center_(1);

                float pixel_xc = p_pixel_yc;
                float pixel_yc = 640 - p_pixel_xc + 1;

                int height = pixel_width;
                int width = pixel_width / 2;
                int xmin = floor(pixel_xc - width / 2 + 0.5);
                int ymin = floor(pixel_yc - height / 2 + 0.5);

                roi.value.setX(xmin);
                roi.value.setY(ymin);
                roi.value.setW(width);
                roi.value.setH(height);
            }

            msg_clusters->push_back(cluster.getIndices());
            msg_rois->push_back(roi);
        }

        msg::publish<GenericVectorMessage, pcl::PointIndices>(out_clusters_unclassified_, msg_clusters);
        msg::publish<GenericVectorMessage, RoiMessage>(out_rois_unclassified_, msg_rois);

        CvMatMessage::Ptr msg_image = std::make_shared<CvMatMessage>(enc::bgr, cloud->header.frame_id, cloud->header.stamp);
        const pcl::PointCloud<pcl::RGB>& src = detector_.rgb_image();
        msg_image->value = cv::Mat(src.height, src.width, CV_8UC3);
        for (int y = 0; y < src.height; ++y)
            for (int x = 0; x < src.width; ++x)
            {
                const pcl::RGB& from = src.at(x, y);
                cv::Vec3b& to = msg_image->value.at<cv::Vec3b>(y, x);

                to[0] = from.b;
                to[1] = from.g;
                to[2] = from.r;
            }

        msg::publish(out_image_, msg_image);
    }
}
