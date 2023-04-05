#include <iostream>
#include <flags/include/flags.h>
#include "util.h"
#include "cppraisr.h"

void print_help()
{
    std::cout << "usage: train [-help] [-q QMatrix] [-v VVector] [-max Max Images] [-threads Number of Threads]" << std::endl;
    std::cout << "arguments:" << std::endl;
    std::cout << "\t-help\tshow this help" << std::endl;
    std::cout << "\t-q\tcontinue to train Q matrix from this file" << std::endl;
    std::cout << "\t-v\tcontinue to train V vector from this file" << std::endl;
    std::cout << "\t-max\tmax number of training images" << std::endl;
    std::cout << "\t-threads\tnumber of threads to use" << std::endl;
}

int main(int argc, char** argv)
{
    using namespace cppraisr;
    const flags::args args(argc, argv);
    if(args.get<bool>("help", false)){
        print_help();
        return 0;
    }

    int32_t max_images = 2147483647;
    int32_t max_threads = -1;
    std::filesystem::path path_q;
    std::filesystem::path path_v;
    std::filesystem::path path_o;
    {
        std::optional<int32_t> option_images = args.get<int32_t>("max");
        if(option_images){
            max_images = option_images.value();
        }
        std::optional<int32_t> option_threads = args.get<int32_t>("threads");
        if(option_threads){
            max_threads = option_threads.value();
        }
        max_threads = std::max(max_threads, 1);

        std::optional<std::string> option_q = args.get<std::string>("q");
        if(option_q){
            path_q = std::filesystem::current_path();
            path_q.append(option_q.value());
        }
        std::optional<std::string> option_v = args.get<std::string>("v");
        if(option_v){
            path_v = std::filesystem::current_path();
            path_v.append(option_v.value());
        }

        std::optional<std::string> option_o = args.get<std::string>("o");
        if(option_o){
            path_o = std::filesystem::current_path();
            path_o.append(option_o.value());
        }
    }

    std::vector<std::filesystem::path> files = parse_directory("train_data",
                                                               [](const std::filesystem::directory_entry& entry) {
                                                                   std::filesystem::path extension = entry.path().extension();
                                                                   if(extension == ".jpg") {
                                                                       return true;
                                                                   }
                                                                   if(extension == ".png") {
                                                                       return true;
                                                                   }
                                                                   return false;
                                                               }, true);
    std::unique_ptr<RAISRTrainer> trainer = std::make_unique<RAISRTrainer>();
    trainer->train(files, path_q, path_v, path_o, max_images);
    return 0;
}
