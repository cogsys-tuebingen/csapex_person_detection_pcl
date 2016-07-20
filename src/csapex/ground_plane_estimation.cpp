#include "ground_plane_estimation.hpp"

#include <csapex/msg/io.h>
#include <csapex/msg/input.h>
#include <csapex/msg/output.h>
#include <csapex/model/node_modifier.h>
#include <csapex/param/parameter_factory.h>
#include <csapex/utility/register_apex_plugin.h>

#include <csapex/msg/generic_vector_message.hpp>

#include <pcl/sample_consensus/sac_model_plane.h>

CSAPEX_REGISTER_CLASS(person_detection::GroundPlaneEstimation, csapex::Node)

using namespace csapex;
using namespace csapex::connection_types;
using namespace person_detection;

GroundPlaneEstimation::GroundPlaneEstimation() :
    initialize_(false)
{

}

void GroundPlaneEstimation::setupParameters(Parameterizable& parameters)
{
    addParameter(param::ParameterFactory::declareTrigger("initialize",
                                                         param::ParameterDescription("Get initial ground estimation based on the selected strategy.")),
                 [this](csapex::param::Parameter*) { initialize_ = true; });

    addParameter(param::ParameterFactory::declareParameterSet<int>("strategy",
                                                                   param::ParameterDescription("Intialization strategy"),
                                                                   {
                                                                       {"naive", Strategy::NAIVE},
                                                                       {"height", Strategy::HEIGHT},
                                                                   },
                                                                   Strategy::NAIVE),
                 [this](csapex::param::Parameter* param) { strategy_ = static_cast<Strategy>(param->as<int>()); });

    auto cond_height = [this]() { return readParameter<int>("strategy") == Strategy::HEIGHT; };
    parameters.addConditionalParameter(param::ParameterFactory::declareRange("height/value",
                                                                             param::ParameterDescription("z/height offset for ground plane"),
                                                                             0.0, 10.0, 0.0, 0.01),
                                       cond_height);
}

void GroundPlaneEstimation::setup(NodeModifier& node_modifier)
{
    in_cloud_ = node_modifier.addInput<PointCloudMessage>("PointCloud");

    out_ground_ = node_modifier.addOutput<GenericVectorMessage, double>("Ground Plane");
}

void GroundPlaneEstimation::process()
{
    if (initialize_)
    {
        PointCloudMessage::ConstPtr pcl_msg(msg::getMessage<PointCloudMessage>(in_cloud_));
        boost::apply_visitor(PointCloudMessage::Dispatch<GroundPlaneEstimation>(this, pcl_msg), pcl_msg->value);
    }
    else
    {
        // nop
    }
}

namespace
{
template<typename PointT>
void select_naive(typename pcl::PointCloud<PointT>::ConstPtr cloud, Eigen::VectorXf& ground_coeffs)
{
    using PointCloudT = typename pcl::PointCloud<PointT>;
    using PointCloudTPtr = typename PointCloudT::Ptr;

    PointCloudTPtr samples(new PointCloudT());

    for (int i = 0; i < 3; ++i)
    {
        const PointT* pt;
        do
        {
            int c = rand() % cloud->width;
            int r = rand() % cloud->height;
            pt = &(cloud->at(c, r));
        } while (std::abs(pt->z) > 0.1);

        samples->push_back(*pt);
    }

    std::vector<int> sample_indices;
    int n = 0;
    std::generate_n(std::back_inserter(sample_indices), samples->size(), [&n] { return n++; });

    pcl::SampleConsensusModelPlane<PointT> model_plane(samples);
    model_plane.computeModelCoefficients(sample_indices, ground_coeffs);
}

template<typename PointT>
void select_height(typename pcl::PointCloud<PointT>::ConstPtr cloud, Eigen::VectorXf& ground_coeffs,
                   double height)
{
    ground_coeffs[0] = 0.0;
    ground_coeffs[1] = -1.0;
    ground_coeffs[2] = 0.0;
    ground_coeffs[3] = height;
}

}

template <class PointT>
void GroundPlaneEstimation::inputCloud(typename pcl::PointCloud<PointT>::ConstPtr cloud)
{
    Eigen::VectorXf ground_coeffs;
    ground_coeffs.resize(4);

    switch (strategy_)
    {
    case Strategy::NAIVE:
        select_naive<PointT>(cloud, ground_coeffs);
        break;
    case Strategy::HEIGHT:
        select_height<PointT>(cloud, ground_coeffs, readParameter<double>("height/value"));
        break;
    }

    std::shared_ptr<std::vector<double>> ground_msg = std::make_shared<std::vector<double>>();
    for (int i = 0; i < 4; ++i)
        ground_msg->push_back(ground_coeffs[i]);

//    std::cout << "Ground: "
//              << "n_x = " << ground_coeffs[0] << " "
//              << "n_y = " << ground_coeffs[1] << " "
//              << "n_z = " << ground_coeffs[2] << " "
//              <<   "d = " << ground_coeffs[3]
//              << std::endl;

    msg::publish<GenericVectorMessage, double>(out_ground_, ground_msg);
}
