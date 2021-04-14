#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include <random>
#include <cmath>
#include <algorithm>
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>



double distance(const pcl::PointXYZ &p1,const pcl::PointXYZ &p2){
    return sqrt(pow(p1.x-p2.x,2) + pow(p1.y-p2.y,2) + pow(p1.z-p2.z,2));
}

void KMeans(const pcl::PointCloud<pcl::PointXYZ>::Ptr pt,const int k = 3){
    const size_t num_iter = 100;

    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<size_t> indices(0, pt->points.size() - 1);

    std::vector<pcl::PointXYZ> means;
    for(size_t i=0;i<k;++i){
        means.push_back(pt->points[indices(random_number_generator)]);
    }

    
    std::vector<size_t> labels(pt->points.size());
    for(size_t i=0;i<num_iter;++i){
        for(size_t p=0;p<pt->points.size();++p){
            double best_dist = std::numeric_limits<double>::max();
            int best_cluster = 10;
            for(size_t cluster=0;cluster<k;++cluster){
                double dist = distance(pt->points[p],means[cluster]);
                if(dist<best_dist){
                    best_dist = dist;
                    best_cluster = cluster;
                }
            }
            labels[p] = best_cluster;
        }
    }

    std::vector<size_t> counts(k,0);
    for(size_t p=0;p<pt->points.size();++p){
        const auto cluster = labels[p];
        means[cluster].x += pt->points[p].x;
        means[cluster].y += pt->points[p].y;
        means[cluster].z += pt->points[p].z;
        counts[cluster]+=1;
    }

    for(size_t cluster=0;cluster<k;++cluster){
        means[cluster].x /= counts[cluster];
        means[cluster].y /= counts[cluster];
        means[cluster].z /= counts[cluster];
    }


    // visualization of k-means
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

    for(size_t cluster=0;cluster<k;++cluster){
        const pcl::PointCloud<pcl::PointXYZ>::Ptr res(new pcl::PointCloud<pcl::PointXYZ>);
        for(size_t i=0;i<pt->points.size();++i){ 
            if(labels[i]==cluster){
                res->points.push_back(pt->points[i]);
            }
        }
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(res,int(255/k*(cluster+1)), int(255/k*(cluster+1)), 150); 
        viewer->addPointCloud<pcl::PointXYZ>(res, single_color,"scene"+std::to_string(cluster));  
        
    }

    // viewer.showCloud(ground,"ground");
    while(!viewer->wasStopped()){
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}   

void visualization(pcl::PointCloud<pcl::PointXYZ>::Ptr points){
    //show point cloud
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(points,0,255,0); 
    viewer->addPointCloud<pcl::PointXYZ>(points, single_color, "sample cloud");

    while(!viewer->wasStopped()){
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

// visualize ground and scene
void visualization(pcl::PointCloud<pcl::PointXYZ>::Ptr points,pcl::PointCloud<pcl::PointXYZ>::Ptr pt){
    //show point cloud
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(points,0,255,0); 
    viewer->addPointCloud<pcl::PointXYZ>(points, single_color1, "sample cloud 1");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(pt,255,0,0); 
    viewer->addPointCloud<pcl::PointXYZ>(pt, single_color2, "sample cloud 2");

    while(!viewer->wasStopped()){
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

// ransac to segment ground
void ransac(const pcl::PointCloud<pcl::PointXYZ>::Ptr pt,pcl::PointCloud<pcl::PointXYZ>::Ptr ground,pcl::PointCloud<pcl::PointXYZ>::Ptr scene){
    float sigma = 0.3;
    size_t num_iter = 100;
    int pre_total_inliner = 0.0;
    
    Eigen::Vector3f best_abc(0.0,0.0,0.0);
    float best_d = 0.0;
    size_t sz = pt->points.size();

    for(size_t i=0;i<num_iter;++i){
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(0,sz-1);
        const size_t index1 = dist(rng);
        size_t index2,index3;
        do{
            index2 = dist(rng);
            index3 = dist(rng);
        }while(index1==index2);

        Eigen::Vector3f p1(pt->points[index1].x,pt->points[index1].y,pt->points[index1].z);
        Eigen::Vector3f p2(pt->points[index2].x,pt->points[index2].y,pt->points[index2].z);
        Eigen::Vector3f p3(pt->points[index3].x,pt->points[index3].y,pt->points[index3].z);

        Eigen::Vector3f v1 = p3-p1;
        Eigen::Vector3f v2 = p2-p1;

        Eigen::Vector3f abc = v1.cross(v2);
        float d = abc.dot(p1);

        int num_inliners = 0;

        for(size_t j=0;j<sz;++j){
            Eigen::Vector3f p(pt->points[j].x,pt->points[j].y,pt->points[j].z);
            if(abs((abc.dot(p)-d)/abc.norm())<sigma){
                num_inliners++;
            }
        }

        if(num_inliners>pre_total_inliner){
            best_abc = abc;
            best_d = d;
            pre_total_inliner = num_inliners;
        }
        
    }

    for(size_t j=0;j<sz;++j){
        Eigen::Vector3f p(pt->points[j].x,pt->points[j].y,pt->points[j].z);
        if(abs((best_abc.dot(p)-best_d)/best_abc.norm())<sigma){
            ground->points.push_back(pt->points[j]);
        }else{
            scene->points.push_back(pt->points[j]);
        }
    }
    
}



int main (int argc, char** argv){
    const std::string in_file("../000000.bin");
    // load point cloud
    std::fstream input(in_file.c_str(), std::ios::in | std::ios::binary);
    if(!input.good()){
        std::cerr << "Could not read file: " << in_file << std::endl;
        exit(EXIT_FAILURE);
    }
    input.seekg(0, std::ios::beg);

    pcl::PointCloud<pcl::PointXYZ>::Ptr points (new pcl::PointCloud<pcl::PointXYZ>);
    for (int i=0; input.good() && !input.eof(); i++) {
        pcl::PointXYZ point;
        input.read((char *) &point.x, 3*sizeof(float));
        char intensity;
        input.read((char *) &intensity, sizeof(float));
        points->push_back(point);
    }

    input.close();

    // visualize pointCloud
    // visualization(points);

    pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
    ransac(points,ground,scene);

    //visualize scene and ground
    // visualization(ground,scene);

    KMeans(scene,5);

    return 0;
}
