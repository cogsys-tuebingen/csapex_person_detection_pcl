#pragma once

#include <csapex/model/node.h>
#include <csapex_point_cloud/point_cloud_message.h>

namespace person_detection
{

class GroundPlaneEstimation : public csapex::Node
{
public:
    enum Strategy { NAIVE, HEIGHT };

    GroundPlaneEstimation();

    virtual void setup(csapex::NodeModifier& node_modifier) override;
    virtual void setupParameters(csapex::Parameterizable &parameters) override;
    virtual void process() override;

    template <class PointT>
    void inputCloud(typename pcl::PointCloud<PointT>::ConstPtr cloud);

private:
    csapex::Input* in_cloud_;

    csapex::Output* out_ground_;

    bool initialize_;
    Strategy strategy_;
};

}
