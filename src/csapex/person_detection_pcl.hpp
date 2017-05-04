#pragma once

#include <csapex/model/node.h>
#include <csapex_point_cloud/msg/point_cloud_message.h>

//#include <person_detection_pcl/ground_based_people_detection_app.h>
#include "../../include/person_detection_pcl/ground_based_people_detection_app.h"

namespace person_detection
{

class PCLPersonDetector : public csapex::Node
{
    using point_t      = pcl::PointXYZRGB;
    using classifier_t = pcl_backport::people::PersonClassifier<::pcl::RGB>;
    using detector_t   = pcl_backport::people::GroundBasedPeopleDetectionApp<point_t>;
    using cluster_t    = pcl_backport::people::PersonCluster<point_t>;
    using cloud_t      = pcl::PointCloud<point_t>;

public:
    PCLPersonDetector();

    virtual void setup(csapex::NodeModifier& node_modifier) override;
    virtual void setupParameters(csapex::Parameterizable &parameters) override;
    virtual void process() override;

    template <class PointT>
    void inputCloud(typename pcl::PointCloud<PointT>::ConstPtr cloud);

    void process(pcl::PointCloud<::pcl::PointXYZI>::ConstPtr cloud);
    void process(pcl::PointCloud<::pcl::PointXYZRGBA>::ConstPtr cloud);
    void process(pcl::PointCloud<::pcl::PointXYZRGBL>::ConstPtr cloud);
    void process(pcl::PointCloud<::pcl::PointXYZRGB>::ConstPtr cloud);

private:
    void update_classifier();
    void update_detector();
    void update_intrinsics();

    void process(cloud_t::Ptr cloud);

private:
    csapex::Input* in_cloud_;
    csapex::Input* in_transfrom_;
    csapex::Input* in_ground_;

    csapex::Output* out_clusters_;
    csapex::Output* out_clusters_euclidean_;
    csapex::Output* out_ground_;
    csapex::Output* out_ground_cloud_;
    csapex::Output* out_no_ground_cloud_;
    csapex::Output* out_clusters_unclassified_;
    csapex::Output* out_rois_unclassified_;
    csapex::Output* out_image_;

    classifier_t classifier_;
    detector_t   detector_;

    Eigen::Matrix3f instrinsics_;
    Eigen::VectorXf ground_plane_;

    struct {
        std::string frame;
        u_int64_t stamp;
    } tmp;

    struct {
        double svm_threshold;
    } param;

};

namespace detail
{
    template<typename PointT, typename = void>
    struct PCLDispatch
    {
        void operator()(PCLPersonDetector* self,
                        typename pcl::PointCloud<PointT>::ConstPtr cloud)
        {
            throw std::runtime_error(std::string("Unsupported PointCloud type: ").append(typeid(PointT).name()));
        }
    };

    template<typename PointT>
    struct PCLDispatch<PointT, typename std::enable_if<std::is_same<PointT, pcl::PointXYZI>::value
                                                        || std::is_same<PointT, pcl::PointXYZRGB>::value
                                                        || std::is_same<PointT, pcl::PointXYZRGBA>::value
                                                        || std::is_same<PointT, pcl::PointXYZRGBL>::value>::type>
    {
        void operator()(PCLPersonDetector* self,
                        typename pcl::PointCloud<PointT>::ConstPtr cloud)
        {
            self->process(cloud);
        }
    };
}

}
