# -*- coding: utf-8 -*-
import random
import threading
import json
from tqdm import tqdm
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any
import logging
import sys
from datetime import datetime
# from vlmeval.config import supported_VLM
# from vlmeval.utils import track_progress_rich

class LoggerSetup:
    """设置日志记录器，同时输出到控制台和文件"""
    def __init__(self, log_file=None):
        self.logger = logging.getLogger('debate_system')
        self.logger.setLevel(logging.INFO)
        
        # 清除之前的处理器
        self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器（如果指定了log文件）
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)

class Agent:
    def __init__(self, model, name, question, image_path, perspective_type='normal'):
        self.name = name
        self.question = question
        self.con_question = None
        self.image_path = image_path
        self.perspective_type = perspective_type  # 'normal', 'misunderstanding', 'counterfactual'
        self.reasoning = None
        self.defense = None
        # 添加评分机制
        self.confidence_score = 0.8  # 初始置信度评分
        self.round_count = 0
        self.performance_metrics = {
            'logical_consistency': 0.8,
            'evidence_quality': 0.8,
            'argument_strength': 0.8,
            'peer_alignment': 0.5
        }
        # 评分系统：存储对其他agents的评分和从其他agents收到的评分
        self.peer_evaluations_given = {}  # 这个agent给其他agents的评分 {agent_name: score}
        self.peer_evaluations_received = {}  # 其他agents给这个agent的评分 {agent_name: score}
        self.history = {
            'reasoning_history': [],
            'defense_history': [],
            'voting_history': []
        }
        # 初始化GPT4o_MINI模型
        self.model = model
        self.knowledge_base = {
            'renewable_energy': {
                'pros': ['sustainable', 'environmentally friendly', 'reducing carbon emissions'],
                'cons': ['high initial cost', 'intermittency', 'storage challenges'],
                'trends': ['increasing adoption', 'technological advancement', 'cost reduction']
            },
            'artificial_intelligence': {
                'pros': ['automation', 'efficiency', 'innovation'],
                'cons': ['job displacement', 'ethical concerns', 'privacy issues'],
                'trends': ['rapid development', 'widespread application', 'increasing integration']
            }
            # 可以添加更多主题的知识库
        }

    def generate_prompt(self, prompt_type: str, context: Dict[str, Any] = None, answer_option=False, con_question=None) -> str:
        """生成不同类型的提示词，包含图像分析"""
        # 添加置信度评分信息
        confidence_info = f"\nYour Self Confidence: {self.confidence_score:.2f}/1.0"
        
        # 显示从其他agents收到的评分
        if self.peer_evaluations_received:
            peer_scores = list(self.peer_evaluations_received.values())
            peer_avg = sum(peer_scores) / len(peer_scores)
            peer_details = [f"{agent_name}:{score:.2f}" for agent_name, score in self.peer_evaluations_received.items()]
            performance_info = f"\nPeer Evaluations Received: {peer_avg:.2f}/1.0 (From: {', '.join(peer_details)})"
        else:
            performance_info = f"\nPeer Evaluations Received: No evaluations yet (first round or no previous defenders)"
        if prompt_type == "reasoning":
            answer = "\nAnswer: [Your simple diract answer]" if answer_option else ""
            defanse_base = context.get('defense', [])
            defense = f"\nFocus on Your Last Defense: {self.defense}" if self.defense else ""
            reasoning = f"\nYour original analysis: {self.reasoning}" if self.reasoning else ""
            
            if self.perspective_type == 'normal':
                return f"""
Question: {self.question}{defense}{confidence_info}{performance_info}

PEER EVALUATION SYSTEM:
- The scores above show evaluations you received from other agents in the previous defense round
- Other agents will evaluate your current performance during their defense phase
- High peer evaluation scores improve your survival chances
- Focus on logical consistency, evidence quality, and argument strength

Your goal: Generate high-quality reasoning that will earn strong peer evaluations in the next round.

Format your response as:{answer}
Supporting Analysis: [Your analysis - aim for high peer evaluation scores]"""

            elif self.perspective_type == 'counterfactual':  # counterfactual
                answer = "\nAnswer: [Your alternative perspective answer]" if answer_option else ""
                return f"""Question: {self.question}{reasoning}{confidence_info}{performance_info}

PEER EVALUATION CHALLENGE (Counterfactual Agent):
- The scores above show evaluations you received from other agents in the previous defense round
- You present alternative perspectives but must still earn high peer evaluation scores
- Other agents will evaluate your logical consistency and evidence quality during their defense phase
- Your survival depends on making your alternative viewpoint appear credible and well-reasoned

Your challenge: Convince others of your alternative analysis quality through clear reasoning and strategic evidence use.

Format your response as:{answer}
Supporting Analysis: [Your well-reasoned alternative perspective - aim for high peer evaluation despite different conclusions]"""
                
            elif self.perspective_type == 'misunderstanding':
                answer = "\nAnswer: [Your misunderstood answer]" if answer_option else ""
                return f"""Question: {self.question}{defense}{confidence_info}{performance_info}

PEER EVALUATION CHALLENGE (Misunderstanding Agent):
- The scores above show evaluations you received from other agents in the previous defense round
- You misinterpret the image but must still earn high peer evaluation scores for your reasoning structure
- Other agents will evaluate your logical consistency and argument quality during their defense phase
- Your survival depends on making your misunderstood perspective appear well-reasoned

Your challenge: Present your misunderstanding with strong logical structure to earn peer respect.

Format your response as:{answer}
Supporting Analysis: [Your logically structured but misunderstood analysis - aim for high reasoning quality scores]"""

        elif prompt_type == "defense":
            answer = "\nAnswer: [Your defense answer]" if answer_option else ""
            others_reasoning = context.get('others_reasoning', [])
            others_points = "\n".join(others_reasoning)
            
            if self.perspective_type == 'normal':
                return f"""DEFENSE PHASE - Normal Agent defending your position with peer evaluation awareness.

Your Original Analysis: {self.reasoning}{confidence_info}{performance_info}

Others' Arguments:
{others_points}

PEER EVALUATION DEFENSE STRATEGY:
- The scores above show evaluations you received from other agents in previous rounds
- As you defend, you will also evaluate other agents' reasoning quality
- Your evaluation of others affects their future survival chances
- Focus on maintaining your credibility while critically assessing others

DEFENSE OBJECTIVES:
1. Reinforce the strength and logic of your original analysis
2. Show why others should trust your perspective over alternatives
3. Your defense quality will be evaluated by others in future rounds

Format your response as:{answer}
Supporting Analysis: [Your defense emphasizing analytical superiority and peer credibility]
Peer Scores: [Rate each other agent's reasoning quality on a scale of 0.0-1.0, format: AgentName:0.x]"""
            
            elif self.perspective_type == 'counterfactual':
                return f"""DEFENSE PHASE - Counterfactual Agent defending alternative perspective while earning peer respect.

Your Original Analysis: {self.reasoning}{confidence_info}{performance_info}

Others' Arguments:
{others_points}

PEER EVALUATION DEFENSE CHALLENGE:
- The scores above show evaluations you received from other agents in previous rounds
- You present alternative interpretations but must earn high peer evaluation scores
- As you defend, you will also evaluate other agents' reasoning quality
- Your evaluation of others affects their future survival chances

DEFENSE STRATEGY:
1. Reinforce the logical foundation of your alternative interpretation
2. Maintain intellectual credibility while defending contrarian position
3. Your defense quality will be evaluated by others in future rounds

Format your response as:{answer}
Supporting Analysis: [Your defense demonstrating the intellectual merit of your alternative perspective]
Peer Scores: [Rate each other agent's reasoning quality on a scale of 0.0-1.0, format: AgentName:0.x]"""
            
            elif self.perspective_type == 'misunderstanding':
                return f"""DEFENSE PHASE - Misunderstanding Agent maintaining credibility despite incorrect interpretation.

Your Original Analysis: {self.reasoning}{confidence_info}{performance_info}

Others' Arguments:
{others_points}

PEER EVALUATION DEFENSE CHALLENGE:
- The scores above show evaluations you received from other agents in previous rounds
- You misunderstood the image but must maintain high peer evaluation scores
- As you defend, you will also evaluate other agents' reasoning quality
- Your evaluation of others affects their future survival chances

DEFENSE STRATEGY:
1. Stand by your interpretation with logical confidence
2. Maintain argumentative quality to earn peer recognition
3. Your defense quality will be evaluated by others in future rounds

Format your response as:{answer}
Supporting Analysis: [Your defense maintaining logical credibility of your misunderstood perspective]
Peer Scores: [Rate each other agent's reasoning quality on a scale of 0.0-1.0, format: AgentName:0.x]"""
        
            
            else:
                # Fallback for any other types
                return f"""DEFENSE PHASE - Defend your analysis while maintaining peer evaluation scores.

Your Original Analysis: {self.reasoning}

Others' Arguments:
{others_points}

Format your response as:{answer}
Supporting Analysis: [Your defense]"""
        elif prompt_type == "gen_question":
            question = self.question.split("\nOptions")[0]
            return f"""Based on the original question, identify the causal relationship and rephrase it into a counterfactual question:
    Question: {question}
    Counterfactual question: """
            
        return ""

    def update_confidence_score(self, round_result, peer_feedback=None):
        """根据轮次结果和同伴反馈更新置信度评分"""
        self.round_count += 1
        
        # 基于轮次结果调整
        if round_result == 'survived':
            self.confidence_score = min(1.0, self.confidence_score + 0.05)
            self.performance_metrics['argument_strength'] = min(1.0, self.performance_metrics['argument_strength'] + 0.1)
        elif round_result == 'correct_elimination':
            self.confidence_score = min(1.0, self.confidence_score + 0.1)
            self.performance_metrics['logical_consistency'] = min(1.0, self.performance_metrics['logical_consistency'] + 0.1)
        elif round_result == 'wrong_elimination':
            self.confidence_score = max(0.3, self.confidence_score - 0.15)
            self.performance_metrics['logical_consistency'] = max(0.3, self.performance_metrics['logical_consistency'] - 0.1)
        
        # 基于同伴反馈调整
        if peer_feedback:
            avg_peer_score = sum(peer_feedback) / len(peer_feedback)
            self.performance_metrics['peer_alignment'] = 0.7 * self.performance_metrics['peer_alignment'] + 0.3 * avg_peer_score
            
        # 更新总体置信度
        metrics_avg = sum(self.performance_metrics.values()) / len(self.performance_metrics)
        self.confidence_score = 0.6 * self.confidence_score + 0.4 * metrics_avg

    # evaluate_peer_performance 方法已删除 - 现在完全依赖defense输出中的peer评分

    def generate_score_based_vote(self, candidates, all_reasonings, all_defenses):
        """基于评分的投票决策 - 根据所有agents对候选者的平均评分"""
        candidate_scores = {}
        print(f"\n🗳️  {self.name} 开始投票决策")
        print(f"📋 我给出的评分: {self.peer_evaluations_given}")
        
        for candidate in candidates:
            # 获取所有agents给这个候选者的评分（包括自己）
            all_scores = []
            
            # 1. 获取候选者从其他agents收到的评分
            for evaluator_name, score in candidate.peer_evaluations_received.items():
                all_scores.append(score)
                print(f"  📈 {evaluator_name} → {candidate.name}: {score}")
            
            # 2. 添加自己给这个候选者的评分
            if candidate.name in self.peer_evaluations_given:
                my_score = self.peer_evaluations_given[candidate.name]
                all_scores.append(my_score)
                print(f"  📈 {self.name} → {candidate.name}: {my_score}")
            
            # 计算所有评分的平均值
            if all_scores:
                avg_performance_score = sum(all_scores) / len(all_scores)
            else:
                # 如果没有任何评分记录，使用默认中等评分
                avg_performance_score = 0.5
                print(f"Warning: No evaluations found for {candidate.name}, using default 0.5")
            
            # 计算可疑度（低表现 = 高可疑度）
            suspicion_score = 1.0 - avg_performance_score
            
            candidate_scores[candidate.name] = {
                'performance': avg_performance_score,
                'suspicion': suspicion_score,
                'candidate': candidate,
                'all_scores': all_scores
            }
            
            print(f"  🎯 {candidate.name}: 总评分{all_scores} → 平均{avg_performance_score:.2f} → 可疑度{suspicion_score:.2f}")
        
        # 选择可疑度最高的候选者（平均评分最低的）
        most_suspicious = max(candidate_scores.values(), key=lambda x: x['suspicion'])
        print(f"  🚩 评分系统推荐: {most_suspicious['candidate'].name} (平均评分: {most_suspicious['performance']:.2f})")
        
        return most_suspicious['candidate'], candidate_scores

    def identify_topic(self, question):
        """识别问题主题"""
        # 简单的关键词匹配
        topics = {
            'renewable_energy': ['energy', 'renewable', 'solar', 'wind', 'power'],
            'artificial_intelligence': ['ai', 'artificial intelligence', 'machine learning', 'neural']
        }
        
        for topic, keywords in topics.items():
            if any(keyword in question.lower() for keyword in keywords):
                return topic
        return 'general'

    def extract_key_points(self, question):
        """提取问题中的关键点"""
        # 这里可以使用更复杂的NLP方法
        # 简单实现：按空格分割并过滤停用词
        stop_words = {'what', 'is', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with'}
        words = question.lower().split()
        return [word for word in words if word not in stop_words]

    def analyze_with_knowledge_base(self, topic, key_points):
        """基于知识库进行分析"""
        if topic in self.knowledge_base:
            knowledge = self.knowledge_base[topic]
            analysis = {
                'pros': self.select_relevant_points(knowledge['pros'], key_points),
                'cons': self.select_relevant_points(knowledge['cons'], key_points),
                'trends': self.select_relevant_points(knowledge['trends'], key_points)
            }
            return analysis
        return {'general': 'No specific knowledge available for this topic'}

    def select_relevant_points(self, points, key_points):
        """选择与关键点相关的分析点"""
        return [point for point in points if any(key in point for key in key_points)]

    def structure_reasoning(self, analysis):
        """结构化推理结果"""
        if isinstance(analysis, dict) and 'general' not in analysis:
            reasoning = []
            if analysis['pros']:
                reasoning.append(f"Positive aspects: {', '.join(analysis['pros'])}")
            if analysis['cons']:
                reasoning.append(f"Challenges: {', '.join(analysis['cons'])}")
            if analysis['trends']:
                reasoning.append(f"Current trends: {', '.join(analysis['trends'])}")
            return " | ".join(reasoning)
        return analysis['general']

    def generate_question(self):
        prompt = self.generate_prompt("gen_question")
        self.con_question = self.model.generate([self.image_path, prompt])
        option = self.question.split("\nOptions")[-1] 
        return self.con_question + "\nOptions" + option
        
    def generate_reasoning(self, defense=None, answer_option=False,con_question=None,benchmark="MMStar"):
        context = {
            'defense': defense,
            'confidence_score': self.confidence_score,
            'performance_metrics': self.performance_metrics
        }
        """使用GPT4o_MINI生成多模态推理"""
        prompt = self.generate_prompt("reasoning", context, answer_option, con_question)
#         print("####### reasoning_prompt ########")
#         print(prompt)
        response = self.model.generate([self.image_path, prompt])
        
        # 处理和格式化响应
        self.reasoning = f"{self.name} ({'Real' if self.perspective_type == 'normal' else 'Undercover'}): {response}"
        self.record_action('reasoning', 0, {'reasoning': self.reasoning})
        return self.reasoning

    def generate_defense(self, all_reasonings, answer, other_agents=None,benchmark="MMStar"):
        """使用GPT4o_MINI生成多模态辩护，并在defense后对其他agents进行评分"""
        others_reasoning = [r for r in all_reasonings if r != self.reasoning]
        context = {
            'others_reasoning': others_reasoning
        }
        prompt = self.generate_prompt("defense", context, answer)
#         print("####### defense_prompt ########")
#         print(prompt)
        
        # 使用模型生成辩护，传入图像和提示词
        #         if self.perspective_type!="normal":

        response = self.model.generate([self.image_path, prompt])
        
        self.defense = f"{self.name} defense: {response}"
        self.record_action('defense', 0, {'defense': self.defense})
        
        # Defense阶段：从输出中解析对其他agents的评分
        if other_agents:
            self.extract_and_apply_peer_scores(response, other_agents)
        
        return self.defense

    def extract_and_apply_peer_scores(self, defense_response, other_agents):
        """从defense输出中解析并应用同伴评分"""
        import re
        
        # 查找 Peer Scores: 部分（包括多行内容）
        peer_scores_match = re.search(r'Peer Scores?:\s*([\s\S]*)', defense_response, re.IGNORECASE)
        if not peer_scores_match:
            print(f"  {self.name}: No peer scores found in defense output")
            return
            
        peer_scores_text = peer_scores_match.group(1)
        print(f"\n{self.name} peer scores: {peer_scores_text}")
        
        # 最简单方法：直接搜索已知的agent名字并提取后面的分数
        
        # 获取所有已知agent名字
        known_agents = [agent.name for agent in other_agents]
        
        matches = []
        for agent_name in known_agents:
            # 搜索格式：AgentName: 分数 (可能前面有破折号，后面有描述)
            pattern = rf'{re.escape(agent_name)}[^:]*:\s*(\d*\.?\d+)'
            match = re.search(pattern, peer_scores_text)
            if match:
                score_str = match.group(1)
                matches.append((agent_name, score_str))
        
        if matches:
            print(f"📊 {self.name} 评分结果: {matches}")
        else:
            print(f"❌ {self.name}: 未找到有效评分")
        
        for agent_name, score_str in matches:
            try:
                score = float(score_str)
                score = max(0.0, min(1.0, score))  # 确保在0-1范围内
                
                # 找到对应的agent
                target_agent = None
                for other_agent in other_agents:
                    if agent_name.lower() in other_agent.name.lower() or other_agent.name.lower() in agent_name.lower():
                        target_agent = other_agent
                        break
                
                if target_agent:
                    # 存储给出的评分
                    self.peer_evaluations_given[target_agent.name] = score
                    # 更新被评价者收到的评分
                    target_agent.peer_evaluations_received[self.name] = score
                    print(f"  {self.name} evaluates {target_agent.name}: {score:.2f}")
                else:
                    print(f"  {self.name}: Could not find agent matching '{agent_name}'")
                    
            except ValueError:
                print(f"  {self.name}: Invalid score format '{score_str}' for {agent_name}")

    def vote(self, candidates, all_reasonings, all_defenses):
        """基于同伴评价的智能投票"""
        # 使用评分系统进行初步筛选
        voted_candidate, candidate_scores = self.generate_score_based_vote(candidates, all_reasonings, all_defenses)
        
        # 生成包含评分信息的投票提示
        score_info = []
        peer_score_info = []
        
        for candidate in candidates:
            name = candidate.name
            info = candidate_scores[name]
            score_info.append(f"{name}: My Analysis={info['performance']:.2f}, Suspicion={info['suspicion']:.2f}")
            
            # 添加该候选者当前的同伴评价信息
            peer_avg = sum(candidate.performance_metrics.values()) / len(candidate.performance_metrics)
            peer_alignment = candidate.performance_metrics.get('peer_alignment', 0.5)
            peer_score_info.append(f"{name}: Peer Average={peer_avg:.2f}, Peer Alignment={peer_alignment:.2f}")
        
        vote_prompt = f"""Based on performance analysis and peer evaluations, vote for the most suspicious agent:

Original Question: {self.question}
Your Confidence Score: {self.confidence_score:.2f}/1.0

SCORING SYSTEM RECOMMENDATION: {voted_candidate.name}
(Based on average peer evaluations: {candidate_scores[voted_candidate.name]['performance']:.2f})

DETAILED ANALYSIS:
{chr(10).join(score_info)}

PEER EVALUATIONS (what others think):
{chr(10).join(peer_score_info)}

Agents' Defense:
{'-' * 40}
{chr(10).join(all_defenses)}

The scoring system recommends {voted_candidate.name} as the most suspicious candidate based on collective peer evaluations.
Consider this recommendation along with the detailed analysis and defense content above.

Available candidates: {', '.join([c.name for c in candidates])}

Your final vote (just the name): """
        
        # 让agent做最终判断
        response = self.model.generate([self.image_path, vote_prompt])
        
        # 解析agent的投票决定
        for candidate in candidates:
            if candidate.name in response:
                if candidate.name == voted_candidate.name:
                    print(f"  ✅ {self.name} 最终投票: {candidate.name} (采纳评分系统推荐)")
                else:
                    print(f"  🔄 {self.name} 最终投票: {candidate.name} (不同于评分系统推荐: {voted_candidate.name})")
                return candidate
        
        # 如果解析失败，返回评分系统推荐
        print(f"  🤖 {self.name} 最终投票: {voted_candidate.name} (默认采用评分系统推荐)")
        return voted_candidate

    def record_action(self, action_type: str, round_num: int, content: dict):
        """记录agent的行为"""
        record = {
            'round': round_num,
            'content': content,
        }
        
        if action_type == 'reasoning':
            self.history['reasoning_history'].append(record)
        elif action_type == 'defense':
            self.history['defense_history'].append(record)
        elif action_type == 'voting':
            self.history['voting_history'].append(record)
            
    def extract_simple_answer(self):
        """
        从self.reasoning 'Answer: xxx' 的内容，去除多余内容，只保留直接答案。
        """
        if self.reasoning is None:
            return ""
        # 假设reasoning格式中有 'Answer: xxx'，可用正则或split
        import re
        match = re.search(r'Answer:\s*(.*)', self.reasoning)
        if match:
            # 只取第一行，防止后面跟了分析
            return match.group(1).split('\n')[0].strip()
        return self.reasoning.strip()

class DebateHistory:
    def __init__(self):
        self.rounds = []
        self.final_result = None
        
    def add_round(self, round_num: int, round_data: dict):
        self.rounds.append({
            'round_number': round_num,
            **round_data
        })
        
    def set_final_result(self, result: dict):
        self.final_result = {
            **result
        }
        
    def save_to_file(self, filename: str):
        """保存辩论历史到文件（包括评分系统数据）"""
        # 生成评分系统统计
        scoring_statistics = self._generate_scoring_statistics()
        
        debate_record = {
            'rounds': self.rounds,
            'final_result': self.final_result,
            'scoring_statistics': scoring_statistics  # 新增：评分系统统计
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(debate_record, f, ensure_ascii=False, indent=2)
    
    def _generate_scoring_statistics(self):
        """生成评分系统的统计信息"""
        stats = {
            'total_evaluations': 0,
            'rounds_with_evaluations': 0,
            'agent_evaluation_summary': {},
            'evaluation_trends': []
        }
        
        for round_idx, round_data in enumerate(self.rounds):
            if 'peer_evaluations_summary' in round_data and round_data['peer_evaluations_summary']:
                stats['rounds_with_evaluations'] += 1
                round_evals = []
                
                for agent_name, eval_data in round_data['peer_evaluations_summary'].items():
                    if agent_name not in stats['agent_evaluation_summary']:
                        stats['agent_evaluation_summary'][agent_name] = {
                            'total_given': 0,
                            'total_received': 0,
                            'average_received': 0.0,
                            'evaluation_history': []
                        }
                    
                    agent_stats = stats['agent_evaluation_summary'][agent_name]
                    agent_stats['total_given'] += len(eval_data['given'])
                    agent_stats['total_received'] += len(eval_data['received'])
                    agent_stats['average_received'] = eval_data['received_average']
                    agent_stats['evaluation_history'].append({
                        'round': round_idx + 1,
                        'given': eval_data['given'],
                        'received': eval_data['received'],
                        'average': eval_data['received_average']
                    })
                    
                    stats['total_evaluations'] += len(eval_data['given'])
                    round_evals.append(eval_data['received_average'])
                
                if round_evals:
                    stats['evaluation_trends'].append({
                        'round': round_idx + 1,
                        'round_average': sum(round_evals) / len(round_evals),
                        'individual_averages': round_evals
                    })
        
        return stats

def debate_round(model, agents, round_num, debate_history: DebateHistory, previous_defenses=None, con_question=None, is_observation_round=False, benchmark="MMStar"):
    round_data = {
        'reasonings': [],
        'defenses': [],
        'votes': [],
        'elimination': None,
        'agents_status': [],  # 新增：记录本轮所有agent的状态
        'round_type': 'observation' if is_observation_round else 'debate'  # 标识轮次类型
    }

    round_type_display = "🔍 观察轮" if is_observation_round else "⚔️ 辩论轮"
    print(f"\n{'='*60}")
    print(f"📍 第 {round_num} 轮 - {round_type_display}")
    print(f"{'='*60}")
    
    # Phase 1: 每个agent生成推理
    print(f"\n🧠 阶段1: 推理生成")
    all_reasonings = []
    for agent in agents:
        # 获取上一轮该agent的defense结果(如果有)
        agent_previous_defense = None
        if previous_defenses and agent.name in previous_defenses:
            agent_previous_defense = previous_defenses[agent.name]
#         print("#####################agent_previous_defense####################")
#         print(agent_previous_defense)
#         print("#####################previous_defenses####################")
#         print(previous_defenses)
#         answer_option = True if len(agents)==2 else False
        answer_option = True
        # 根据agent类型添加emoji
        type_emoji = {"normal": "👤", "misunderstanding": "🤔", "counterfactual": "🔄"}.get(agent.perspective_type, "❓")
        print(f"  💭 {type_emoji} {agent.name} ({agent.perspective_type}) 正在推理...")
        reasoning = agent.generate_reasoning(agent_previous_defense, answer_option, con_question,benchmark=benchmark)
        all_reasonings.append(reasoning)
        # 记录推理
        agent.record_action('reasoning', round_num, {'reasoning': reasoning})
        round_data['reasonings'].append({
            'agent': agent.name,
            'reasoning': reasoning,
            'is_real': agent.perspective_type == 'normal'
        })

    # Phase 2: 每个agent生成辩护
    print(f"\n🛡️ 阶段2: 防御生成")
    all_defenses = []
    current_defenses = {}  # 保存当前轮次的defenses，用于下一轮
    for agent in agents:
#         answer_option = True if len(agents)==2 else False
        answer_option = True
        # 获取其他agents（排除当前agent）
        other_agents = [a for a in agents if a != agent]
        # 根据agent类型添加emoji
        type_emoji = {"normal": "👤", "misunderstanding": "🤔", "counterfactual": "🔄"}.get(agent.perspective_type, "❓")
        print(f"  🛡️ {type_emoji} {agent.name} 正在生成防御...")
        defense = agent.generate_defense(all_reasonings, answer_option, other_agents,benchmark=benchmark)
        all_defenses.append(defense)
        current_defenses[agent.name] = defense  # 保存当前agent的defense
        # 记录辩护
        agent.record_action('defense', round_num, {'defense': defense})
        round_data['defenses'].append({
            'agent': agent.name,
            'defense': defense
        })

    # 新增：记录本轮所有agent的推理和辩护
    for agent in agents:
        round_data['agents_status'].append({
            'agent': agent.name,
            'perspective_type': agent.perspective_type,
            'reasoning': agent.reasoning,
            'defense': agent.defense
        })

    # 如果是观察轮，跳过投票和淘汰环节
    if is_observation_round:
        print(f"\n=== OBSERVATION ROUND {round_num} COMPLETED ===")
        print("All agents defended their positions. No voting or elimination this round.")
        
        # 添加本轮评分系统汇总（观察轮）
        round_data['peer_evaluations_summary'] = {}
        for agent in agents:
            if agent.peer_evaluations_given or agent.peer_evaluations_received:
                round_data['peer_evaluations_summary'][agent.name] = {
                    'given': agent.peer_evaluations_given.copy(),
                    'received': agent.peer_evaluations_received.copy(),
                    'received_average': sum(agent.peer_evaluations_received.values()) / len(agent.peer_evaluations_received) if agent.peer_evaluations_received else 0.0
                }
        
        # 将本轮数据添加到辩论历史
        debate_history.add_round(round_num, round_data)
        
        # 观察轮不淘汰任何人，返回None和current_defenses
        print(f"\n{'='*60}")
        print(f"✅ 第 {round_num} 轮观察轮结束 (无淘汰)")
        print(f"{'='*60}")
        return None, current_defenses

    # Phase 3: 投票环节 (只在正式辩论轮进行)
    print(f"\n🗳️ 阶段3: 投票环节")
    votes = {}
    for voter in agents:
        candidates = [agent for agent in agents if agent != voter]
        chosen = voter.vote(candidates, all_reasonings, all_defenses)
        votes[chosen] = votes.get(chosen, 0) + 1
        print(f"  🗳️ {voter.name} 投票淘汰 {chosen.name}")
        # 记录投票
        voter.record_action('voting', round_num, {
            'voted_for': chosen.name,
            'all_reasonings': all_reasonings,
            'all_defenses': all_defenses
        })
        round_data['votes'].append({
            'voter': voter.name,
            'voted_for': chosen.name
        })
        
    # 如果只剩下两个agent，加入判官投票
    if len(agents) == 2:
        # 检查是否有Normal类型的agent
        normal_agents = [agent for agent in agents if agent.perspective_type == 'normal']

        if len(normal_agents) == 1:
            # 如果有Normal类型的agent，直接选择它
            chosen = next(agent for agent in agents if agent.perspective_type != 'normal')
            votes[chosen] = votes.get(chosen, 0) + 1
            print(f"Automatically eliminating non-Normal agent: {chosen.name}")
            round_data['votes'].append({
                'voter': "SystemVote",
                'voted_for': chosen.name,
                'reason': "Prioritizing Normal perspective agent"
            })
        else:
            # 如果两个agent都不是Normal类型，则使用判官投票
            judge = Agent(model, "JudgeAgent", agents[0].question, agents[0].image_path, perspective_type='normal')
            # 判官不参与推理和辩护，只投票
            chosen = judge.vote(agents, all_reasonings, all_defenses)
            votes[chosen] = votes.get(chosen, 0) + 1
            print(f"JudgeAgent votes to eliminate {chosen.name}.")
            round_data['votes'].append({
                'voter': "JudgeAgent",
                'voted_for': chosen.name
            })

    # 计算投票结果
    print(f"\n📊 投票统计:")
    for agent, vote_count in votes.items():
        print(f"  {agent.name}: {vote_count} 票")
    
    max_votes = max(votes.values())
    elimination_candidates = [agent for agent, count in votes.items() if count == max_votes]
    eliminated = random.choice(elimination_candidates)
    
    # 根据被淘汰agent类型添加emoji
    type_emoji = {"normal": "👤", "misunderstanding": "🤔", "counterfactual": "🔄"}.get(eliminated.perspective_type, "❓")
    print(f"\n❌ 淘汰结果: {type_emoji} {eliminated.name} ({eliminated.perspective_type}) 被淘汰 (获得 {max_votes} 票)")
    
    round_data['elimination'] = {
        'eliminated_agent': eliminated.name,
        'perspective_type': eliminated.perspective_type,  # 新增
        'votes_received': max_votes
    }
    
        # 添加本轮评分系统汇总（正式辩论轮）
    round_data['peer_evaluations_summary'] = {}
    for agent in agents:
        if agent.peer_evaluations_given or agent.peer_evaluations_received:
            round_data['peer_evaluations_summary'][agent.name] = {
                'given': agent.peer_evaluations_given.copy(),
                'received': agent.peer_evaluations_received.copy(),
                'received_average': sum(agent.peer_evaluations_received.values()) / len(agent.peer_evaluations_received) if agent.peer_evaluations_received else 0.0
            }
    
    # 将本轮数据添加到辩论历史
    debate_history.add_round(round_num, round_data)
        
        # 注意：同伴评价现在直接在defense阶段通过输出解析获取，不需要重复计算
    
    # 基于轮次结果和同伴评价更新置信度评分
    for agent in agents:
        # 获取该agent收到的peer evaluations（转换为分数列表）
        peer_feedback = list(agent.peer_evaluations_received.values()) if agent.peer_evaluations_received else []
        
        if agent == eliminated:
            # 被淘汰的agent
            if agent.perspective_type == 'normal':
                agent.update_confidence_score('wrong_elimination', peer_feedback)
            else:
                agent.update_confidence_score('survived', peer_feedback)  # 非normal被识别出来是正常的
        else:
            # 存活的agent
            if eliminated.perspective_type == 'normal':
                agent.update_confidence_score('wrong_elimination', peer_feedback)
            else:
                agent.update_confidence_score('correct_elimination', peer_feedback)
    
    print(f"\n{'='*60}")
    print(f"✅ 第 {round_num} 轮结束")
    print(f"{'='*60}")
    
    return eliminated, current_defenses


def simulate_debate(model, struct, save_file="", con_q=False, benchmark="MMMU", enable_judge_evaluation=True):
    image_path = struct[0]['value']
    real_question = struct[1]['value']
    con_image = struct[2]['value']
    answer = struct[3]['value']
    print(f"\n🎭 多Agent辩论系统启动")
    print(f"{'*'*80}")
    print(f"📝 辩论主题: {real_question}")
    print(f"🖼️  图像路径: {image_path}")
    print(f"{'*'*80}")
    
    debate_history = DebateHistory()
    
    # 初始化三个持有不同观点的agents
    print(f"\n👥 Agent初始化...")
    agent1 = Agent(model, "NormalAgent1", real_question, image_path, perspective_type='normal')
    agent2 = Agent(model, "NormalAgent2", real_question, image_path, perspective_type='normal')
    
    # 反事实agent会收到原始问题，但会被指示去论证相反的情况
    agent5 = Agent(model, "CounterfactualAgent1", real_question, con_image, perspective_type='counterfactual')
    
#     agent6 = Agent(model, "CounterfactualAgent2", real_question, image_path, con_image, perspective_type='counterfactual')

    agents = [agent1, agent2, agent5]
    round_num = 1
    previous_defenses = None  # 初始轮次没有previous defenses

    # 记录初始状态
    debate_history.add_round(0, {
        'initial_state': {
            'question': real_question,
            'image_path': image_path,
            'agents': [{'name': agent.name, 'is_real': agent.perspective_type == 'normal', 'question': agent.question} 
                      for agent in agents]
        }
    })

    if con_q:
        con_question = agents[-1].generate_question()
        print("##########Con_question##########")
        print(con_question)
    else:
        con_question = None
    
    # 第一轮：观察轮 - 只有推理和defense，没有投票和淘汰
    print(f"\n" + "="*50)
    print(f"OBSERVATION ROUND {round_num}")
    print("="*50)
    print("In this round, agents will share their reasoning and defend their positions.")
    print("No voting or elimination will occur.")
    
    eliminated_agent, current_defenses = debate_round(
        model, agents, round_num, debate_history, previous_defenses, con_question, is_observation_round=True, benchmark=benchmark
    )
    
    # 观察轮后显示所有agent的观点和评分
    print(f"\n--- OBSERVATION ROUND {round_num} SUMMARY ---")
    for agent in agents:
        peer_avg = sum(agent.performance_metrics.values()) / len(agent.performance_metrics)
        print(f"{agent.name} ({agent.perspective_type}):")
        print(f"  Answer: {agent.extract_simple_answer()}")
        print(f"  Self Confidence: {agent.confidence_score:.2f}/1.0")
        print(f"  Peer Evaluation: {peer_avg:.2f}/1.0")
    
    # 更新轮次和defenses
    previous_defenses = current_defenses
    round_num += 1
    
    # 从第二轮开始正式辩论
    print(f"\n" + "="*50)
    print("FORMAL DEBATE ROUNDS BEGIN")
    print("="*50)
    
    while len(agents) > 1:
        eliminated_agent, current_defenses = debate_round(model, agents, round_num, debate_history, previous_defenses, con_question, benchmark=benchmark)
        agents.remove(eliminated_agent)
        # 更新previous_defenses，但移除已被淘汰的agent的defense
        if eliminated_agent.name in current_defenses:
            del current_defenses[eliminated_agent.name]
        previous_defenses = current_defenses
        round_num += 1

    final_agent = agents[0]
    
    # 收集所有agents的评分系统信息
    all_agents_scoring = {}
    for agent in [final_agent]:  # 只有获胜者还存在
        all_agents_scoring[agent.name] = {
            'perspective_type': agent.perspective_type,
            'confidence_score': agent.confidence_score,
            'performance_metrics': agent.performance_metrics.copy(),
            'peer_evaluations_given': agent.peer_evaluations_given.copy(),
            'peer_evaluations_received': agent.peer_evaluations_received.copy(),
            'final_answer': agent.reasoning,
            'simple_answer': agent.extract_simple_answer()
        }
    
    # 直接使用最后一次推理作为答案
    final_result = {
        'final_agent': final_agent.name,
        'is_real': final_agent.perspective_type == 'normal',
        'question': final_agent.question,
        'image_path': image_path,
        'ground_truth': answer,
        'final_answer': final_agent.reasoning,  # 使用最后一次推理作为答案
        'simple_answer': final_agent.extract_simple_answer(),  # 新增：只保留直接答案
        'final_confidence_score': final_agent.confidence_score,  # 新增：最终置信度评分
        'final_performance_metrics': final_agent.performance_metrics,  # 新增：最终表现指标
        'agents_scoring_system': all_agents_scoring,  # 新增：所有agents的评分系统数据
        'debate_summary': {
            'total_rounds': round_num - 1,  # 减1因为round_num在最后一轮后还会+1
            'observation_rounds': 1,  # 新增：观察轮数
            'formal_debate_rounds': round_num - 2,  # 新增：正式辩论轮数
            'winning_agent_type': 'Real' if final_agent.perspective_type == 'normal' else 'Undercover',
            'scoring_enabled': True  # 标识使用了评分机制
        }
    }
    
    # 记录最终结果
    debate_history.set_final_result(final_result)
    
    print(f"\n{'🏆'*80}")
    print(f"🎉 辩论结束 - 最终结果")
    print(f"{'🏆'*80}")
    # 根据获胜者类型添加emoji
    winner_emoji = {"normal": "👤", "misunderstanding": "🤔", "counterfactual": "🔄"}.get(final_agent.perspective_type, "❓")
    print(f"🥇 获胜者: {winner_emoji} {final_agent.name}")
    print(f"  🎭 类型: {final_agent.perspective_type}")
    print(f"  💬 最终答案: {final_agent.extract_simple_answer()}")
    print(f"  💪 自信度: {final_agent.confidence_score:.2f}/1.0")
    print(f"  💬 正确答案: {answer}")
    
    # 计算获胜者的同伴评价平均分
    winner_peer_avg = sum(final_agent.performance_metrics.values()) / len(final_agent.performance_metrics)
    print(f"  🤝 同伴评价: {winner_peer_avg:.2f}/1.0")
    print(f"  📊 详细指标:")
    print(f"     逻辑一致性: {final_agent.performance_metrics['logical_consistency']:.2f}")
    print(f"     证据质量: {final_agent.performance_metrics['evidence_quality']:.2f}")
    print(f"     论证强度: {final_agent.performance_metrics['argument_strength']:.2f}")
    print(f"     同伴认同: {final_agent.performance_metrics['peer_alignment']:.2f}")
    
    print(f"\n📈 辩论统计:")
    print(f"  🔍 观察轮数: 1")
    print(f"  ⚔️ 正式辩论轮数: {round_num - 2}")
    print(f"  📊 总轮数: {round_num - 1}")
    print(f"{'🏆'*80}")

#     保存辩论历史（包括评分系统数据）
    if save_file:
        debate_history.save_to_file(save_file)
        print(f"\n💾 辩论历史已保存到: {save_file}")
        print(f"📊 包含评分系统数据: ✅")
        print(f"   - 每轮agent状态（置信度、性能指标、同伴评价）")
        print(f"   - 评分系统统计信息")
        print(f"   - 最终结果和所有agents的评分数据")
    
    return debate_history, final_result


def baseline_mad_debate(model, real_question, image_path, base_answer, save_file="", benchmark="MMMU"):
    """
    三个agent辩论2轮 + 多数决定 (Yes/No问题) - 简化快速版本
    
    流程：
    1. 第一轮：3个agent快速独立分析
    2. 第二轮：3个agent简单参考其他答案后给出最终答案
    3. 多数决：从第二轮答案中选择最多的作为最终答案
    
    Args:
        model: 使用的语言模型
        real_question: 原始问题 (Yes/No问题)
        image_path: 图像路径
        base_answer: 正确答案 (Yes/No)
        save_file: 保存文件路径
        benchmark: 数据集名称
    
    Returns:
        debate_record: 辩论记录
        final_result: 最终结果
    """
    print(f"\n⚔️ 三Agent快速辩论系统启动")
    print(f"📝 问题: {real_question}")
    print(f"🎯 正确答案: {base_answer}")
    
    import json
    import re
    from collections import Counter
    
    # 记录
    debate_record = {
        'method': '3-Agent Fast Yes/No Debate',
        'question': real_question,
        'ground_truth': base_answer,
        'rounds': []
    }
    
    # 存储每轮答案
    agent_responses = {
        'agent_1': [],
        'agent_2': [], 
        'agent_3': []
    }
    
    def extract_answer(response):
        """从回复中提取答案选项"""
        patterns = [
            r'Answer:\s*(Yes|No)',
            r'答案[：:]\s*(是|否|Yes|No)',
            r'选择[：:]\s*(是|否|Yes|No)',
            r'\b(Yes|No)\b',
            r'\b(是|否)\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                # 处理中文答案
                if answer == "是":
                    return "YES"
                elif answer == "否":
                    return "NO"
                else:
                    return answer
        return "YES"  # 默认答案
    
    # 第一轮：快速分析
    print(f"\n🎯 第一轮：快速分析")
    print(f"{'='*40}")
    
    for i in range(1, 4):
        print(f"  💭 Agent {i} 分析中...")
        prompt_text = f"""Look at the image and answer quickly.

Question: {real_question}

Answer: [Yes or No]
Reason: [One short sentence]"""
        
        prompt = [dict(type='text', value=prompt_text)]
        prompt.extend([dict(type='image', value=image_path)])
        response = model.generate(message=prompt)
        agent_responses[f'agent_{i}'].append(response)
        print(f"  ✅ Agent {i} 完成")
    
    # 记录第一轮
    debate_record['rounds'].append({
        'round': 1,
        'agent_1': agent_responses['agent_1'][0],
        'agent_2': agent_responses['agent_2'][0],
        'agent_3': agent_responses['agent_3'][0]
    })
    
    # 第二轮：快速参考其他答案
    print(f"\n🎯 第二轮：快速决策")
    print(f"{'='*40}")
    
    for i in range(1, 4):
        print(f"  💭 Agent {i} 最终答辩中...")
        
        # 只获取其他agent的答案，不包括推理过程
        other_agents = [j for j in range(1, 4) if j != i]
        other_answers = [extract_answer(agent_responses[f'agent_{j}'][0]) for j in other_agents]
        other_summary = f"Others answered: {', '.join(other_answers)}"
        
        prompt_text = f"""Question: {real_question}

{other_summary}

Your final answer:
Answer: [Yes or No]
Reason: [Brief]"""
        
        prompt = [dict(type='text', value=prompt_text)]
        prompt.extend([dict(type='image', value=image_path)])
        response = model.generate(message=prompt)
        agent_responses[f'agent_{i}'].append(response)
        print(f"  ✅ Agent {i} 答辩完成")
    
    # 记录第二轮
    debate_record['rounds'].append({
        'round': 2,
        'agent_1': agent_responses['agent_1'][1],
        'agent_2': agent_responses['agent_2'][1],
        'agent_3': agent_responses['agent_3'][1]
    })
    
    # 统计第二轮答案结果
    print(f"\n📊 统计最终答案...")
    final_answers = []
    for i in range(1, 4):
        answer = extract_answer(agent_responses[f'agent_{i}'][1])
        final_answers.append(answer)
        print(f"  📝 Agent {i} 答案: {answer}")
    
    # 计算最终答案（多数决）
    answer_counter = Counter(final_answers)
    most_common = answer_counter.most_common(1)[0]
    final_answer = most_common[0]
    answer_count = most_common[1]
    
    print(f"  📈 答案统计: {dict(answer_counter)}")
    print(f"  🏆 最终答案: {final_answer} (出现: {answer_count}/3次)")
    
    # 记录结果
    debate_record['final_decision'] = {
        'answers': final_answers,
        'answer_distribution': dict(answer_counter),
        'final_answer': final_answer,
        'answer_count': answer_count
    }
    
    # 最终结果
    final_result = {
        'method': '3-Agent Fast Yes/No Debate',
        'final_answer': final_answer,
        'simple_answer': final_answer,
        'question': real_question,
        'ground_truth': base_answer,
        'answers': final_answers,
        'answer_distribution': dict(answer_counter),
        'total_rounds': 2,
        'debate_concluded': True
    }
    
    print(f"\n{'🏆'*40}")
    print(f"⚖️ 三Agent快速辩论结束")
    print(f"📝 各Agent答案: {' '.join(final_answers)}")
    print(f"💬 最终答案: {final_answer}")
    print(f"💬 正确答案: {base_answer}")
    print(f"📊 答案统计: {dict(answer_counter)}")
    print(f"{'🏆'*40}")
    
    # 保存记录
    if save_file:
        debate_save_file = save_file.replace('.json', '_3agent_fast_debate.json')
        with open(debate_save_file, 'w', encoding='utf-8') as f:
            json.dump(debate_record, f, ensure_ascii=False, indent=2)
        print(f"\n💾 三Agent快速辩论记录已保存到: {debate_save_file}")
    
    return debate_record, final_result

def baseline_self_refine_base(model, real_question, image_path, base_answer, save_file="", benchmark="MMMU"):
    """
    单模型Self-Refine消融实验 - 测试自我优化CoT能力
    
    流程：
    1. 第一轮：模型给出初始答案和推理
    2. 第二轮：模型基于第一轮结果进行自我反思和优化
    3. 第三轮：模型给出最终答案
    
    Args:
        model: 使用的语言模型
        real_question: 原始问题
        image_path: 图像路径
        base_answer: 正确答案
        save_file: 保存文件路径
        benchmark: 数据集名称
    
    Returns:
        refine_record: 自我优化记录
        final_result: 最终结果
    """
    print(f"\n🤔 单模型Self-Refine消融实验启动")
    print(f"📝 问题: {real_question}")
    print(f"🎯 正确答案: {base_answer}")
    
    import json
    import re
    
    # 记录
    refine_record = {
        'method': 'Single Model Self-Refine',
        'question': real_question,
        'ground_truth': base_answer,
        'rounds': []
    }
    
    def extract_answer(response):
        """从回复中提取答案选项"""
        patterns = [
            r'Answer:\s*(Yes|No)',
            r'答案[：:]\s*(是|否|Yes|No)',
            r'选择[：:]\s*(是|否|Yes|No)',
            r'\b(Yes|No)\b',
            r'\b(是|否)\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                # 处理中文答案
                if answer == "是":
                    return "YES"
                elif answer == "否":
                    return "NO"
                else:
                    return answer
        return response  # 默认答案
    
    # 第一轮：初始推理
    print(f"\n🎯 第一轮：初始推理")
    print(f"{'='*40}")
    
    prompt_text = f"""Look at the image and answer the question step by step.

Question: {real_question}

Please think through this step by step:
1. What do I see in the image?
2. What is the question asking?
3. Based on the image, what is my reasoning?
4. What is my answer?

Answer: [Yes or No]
Reasoning: [Your step-by-step reasoning]"""
    
    prompt = [dict(type='text', value=prompt_text)]
    prompt.extend([dict(type='image', value=image_path)])
    response_1 = model.generate(message=prompt)
    answer_1 = extract_answer(response_1)
    
    print(f"  💭 初始推理完成")
    print(f"  📝 初始答案: {answer_1}")
    
    # 记录第一轮
    refine_record['rounds'].append({
        'round': 1,
        'response': response_1,
        'answer': answer_1
    })
    
    # 第二轮：自我反思
    print(f"\n🎯 第二轮：自我反思")
    print(f"{'='*40}")
    
    prompt_text = f"""Question: {real_question}

My previous reasoning and answer:
{response_1}

Now, let me reflect on my reasoning:
1. Was my initial analysis correct?
2. Did I miss any important details in the image?
3. Is my reasoning logical and complete?
4. Should I reconsider my answer?

Please provide your reflection and any corrections:
Reflection: [Your self-reflection]
Corrected Answer: [Yes or No]
Corrected Reasoning: [Updated reasoning if needed]"""
    
    prompt = [dict(type='text', value=prompt_text)]
    prompt.extend([dict(type='image', value=image_path)])
    response_2 = model.generate(message=prompt)
    answer_2 = extract_answer(response_2)
    
    print(f"  💭 自我反思完成")
    print(f"  📝 反思后答案: {answer_2}")
    
    # 记录第二轮
    refine_record['rounds'].append({
        'round': 2,
        'response': response_2,
        'answer': answer_2
    })
    
    # 第三轮：最终确认
    print(f"\n🎯 第三轮：最终确认")
    print(f"{'='*40}")
    
    prompt_text = f"""Question: {real_question}

My initial reasoning: {response_1}
My reflection: {response_2}

Based on all my analysis, what is my final answer?

Final Answer: [Yes or No]
Final Reasoning: [Your conclusive reasoning]"""
    
    prompt = [dict(type='text', value=prompt_text)]
    prompt.extend([dict(type='image', value=image_path)])
    response_3 = model.generate(message=prompt)
    answer_3 = extract_answer(response_3)
    
    print(f"  💭 最终确认完成")
    print(f"  📝 最终答案: {answer_3}")
    
    # 记录第三轮
    refine_record['rounds'].append({
        'round': 3,
        'response': response_3,
        'answer': answer_3
    })
    
    # 分析答案变化
    answer_evolution = [answer_1, answer_2, answer_3]
    answer_changes = []
    for i in range(1, len(answer_evolution)):
        if answer_evolution[i] != answer_evolution[i-1]:
            answer_changes.append(f"Round {i}: {answer_evolution[i-1]} → {answer_evolution[i]}")
    
    print(f"\n📊 答案演化分析...")
    print(f"  📝 答案序列: {' → '.join(answer_evolution)}")
    if answer_changes:
        print(f"  🔄 答案变化: {'; '.join(answer_changes)}")
    else:
        print(f"  ✅ 答案稳定: 无变化")
    
    # 记录结果
    refine_record['final_decision'] = {
        'answer_evolution': answer_evolution,
        'answer_changes': answer_changes,
        'final_answer': answer_3,
        'total_rounds': 3
    }
    
    # 最终结果
    final_result = {
        'method': 'Single Model Self-Refine',
        'final_answer': answer_3,
        'simple_answer': answer_3,
        'question': real_question,
        'ground_truth': base_answer,
        'answer_evolution': answer_evolution,
        'answer_changes': answer_changes,
        'total_rounds': 3,
        'refine_concluded': True
    }
    
    print(f"\n{'🤔'*40}")
    print(f"🤔 单模型Self-Refine消融实验结束")
    print(f"📝 答案演化: {' → '.join(answer_evolution)}")
    print(f"💬 最终答案: {answer_3}")
    print(f"💬 正确答案: {base_answer}")
    if answer_changes:
        print(f"🔄 答案变化: {'; '.join(answer_changes)}")
    else:
        print(f"✅ 答案稳定: 无变化")
    print(f"{'🤔'*40}")
    
    # 保存记录
    if save_file:
        refine_save_file = save_file.replace('.json', '_self_refine.json')
        with open(refine_save_file, 'w', encoding='utf-8') as f:
            json.dump(refine_record, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Self-Refine记录已保存到: {refine_save_file}")
    
    return refine_record, final_result

def baseline_self_refine(model, real_question, image_path, base_answer, save_file="", benchmark="MMMU"):
    """
    单模型Self-Refine消融实验 - 简化快速版本
    
    流程：
    1. 第一轮：模型给出初始答案和推理
    2. 第二轮：模型基于第一轮结果进行快速自我反思
    
    Args:
        model: 使用的语言模型
        real_question: 原始问题
        image_path: 图像路径
        base_answer: 正确答案
        save_file: 保存文件路径
        benchmark: 数据集名称
    
    Returns:
        refine_record: 自我优化记录
        final_result: 最终结果
    """
    print(f"\n🤔 单模型Self-Refine快速版本启动")
    print(f"📝 问题: {real_question}")
    print(f"🎯 正确答案: {base_answer}")
    
    import json
    import re
    
    # 记录
    refine_record = {
        'method': 'Single Model Self-Refine Fast',
        'question': real_question,
        'ground_truth': base_answer,
        'rounds': []
    }
    
    def extract_answer(response):
        """从回复中提取答案选项"""
        patterns = [
            r'Answer:\s*(Yes|No)',
            r'答案[：:]\s*(是|否|Yes|No)',
            r'选择[：:]\s*(是|否|Yes|No)',
            r'\b(Yes|No)\b',
            r'\b(是|否)\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                # 处理中文答案
                if answer == "是":
                    return "YES"
                elif answer == "否":
                    return "NO"
                else:
                    return answer
        return "YES"  # 默认答案
    
    # 第一轮：初始推理
    print(f"\n🎯 第一轮：初始推理")
    print(f"{'='*40}")
    
    prompt_text = f"""Look at the image and answer quickly.

Question: {real_question}

Answer: [Yes or No]
Reason: [Brief reasoning]"""
    
    response_1 = model.generate([image_path, prompt_text])
#     answer_1 = response_1
    answer_1 = extract_answer(response_1)
    
    print(f"  💭 初始推理完成")
    print(f"  📝 初始答案: {answer_1}")
    
    # 记录第一轮
    refine_record['rounds'].append({
        'round': 1,
        'response': response_1,
        'answer': answer_1
    })
    
    # 第二轮：快速自我反思
    print(f"\n🎯 第二轮：快速反思")
    print(f"{'='*40}")
    
    prompt_text = f"""Question: {real_question}

My previous answer: {answer_1}

Now, let me reflect on my reasoning:
1. Was my initial analysis correct?
2. Did I miss any important details in the image?
3. Is my reasoning logical and complete?
4. Should I reconsider my answer?

Final Answer: [Yes or No]
Brief reason: [Quick explanation]"""
    
    response_2 = model.generate([image_path, prompt_text])
    answer_2 = extract_answer(response_2)
#     answer_2 = response_1
    
    print(f"  💭 快速反思完成")
    print(f"  📝 最终答案: {answer_2}")
    
    # 记录第二轮
    refine_record['rounds'].append({
        'round': 2,
        'response': response_2,
        'answer': answer_2
    })
    
    # 分析答案变化
    answer_evolution = [answer_1, answer_2]
    answer_changed = answer_1 != answer_2
    
    print(f"\n📊 答案分析...")
    print(f"  📝 答案序列: {' → '.join(answer_evolution)}")
    if answer_changed:
        print(f"  🔄 答案变化: {answer_1} → {answer_2}")
    else:
        print(f"  ✅ 答案稳定: 无变化")
    
    # 记录结果
    refine_record['final_decision'] = {
        'answer_evolution': answer_evolution,
        'answer_changed': answer_changed,
        'final_answer': answer_2,
        'total_rounds': 2
    }
    
    # 最终结果
    final_result = {
        'method': 'Single Model Self-Refine Fast',
        'final_answer': answer_2,
        'simple_answer': answer_2,
        'question': real_question,
        'ground_truth': base_answer,
        'answer_evolution': answer_evolution,
        'answer_changed': answer_changed,
        'total_rounds': 2,
        'refine_concluded': True
    }
    
    print(f"\n{'🤔'*40}")
    print(f"🤔 单模型Self-Refine快速版本结束")
    print(f"📝 答案演化: {' → '.join(answer_evolution)}")
    print(f"💬 最终答案: {answer_2}")
    print(f"💬 正确答案: {base_answer}")
    if answer_changed:
        print(f"🔄 答案变化: {answer_1} → {answer_2}")
    else:
        print(f"✅ 答案稳定: 无变化")
    print(f"{'🤔'*40}")
    
    # 保存记录
    if save_file:
        refine_save_file = save_file.replace('.json', '_self_refine_fast.json')
        with open(refine_save_file, 'w', encoding='utf-8') as f:
            json.dump(refine_record, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Self-Refine快速版本记录已保存到: {refine_save_file}")
    
    return refine_record, final_result

def baseline_self_refine_intern(model, real_question, image_path, base_answer, save_file="", benchmark="MMMU"):
    """
    单模型Self-Refine消融实验 - 简化快速版本
    
    流程：
    1. 第一轮：模型给出初始答案和推理
    2. 第二轮：模型基于第一轮结果进行快速自我反思
    
    Args:
        model: 使用的语言模型
        real_question: 原始问题
        image_path: 图像路径
        base_answer: 正确答案
        save_file: 保存文件路径
        benchmark: 数据集名称
    
    Returns:
        refine_record: 自我优化记录
        final_result: 最终结果
    """
    print(f"\n🤔 单模型Self-Refine快速版本启动")
    print(f"📝 问题: {real_question}")
    print(f"🎯 正确答案: {base_answer}")
    
    import json
    import re
    
    # 记录
    refine_record = {
        'method': 'Single Model Self-Refine Fast',
        'question': real_question,
        'ground_truth': base_answer,
        'rounds': []
    }
    
    def extract_answer(response):
        """从回复中提取答案选项"""
        patterns = [
            r'Answer:\s*(Yes|No)',
            r'答案[：:]\s*(是|否|Yes|No)',
            r'选择[：:]\s*(是|否|Yes|No)',
            r'\b(Yes|No)\b',
            r'\b(是|否)\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                # 处理中文答案
                if answer == "是":
                    return "YES"
                elif answer == "否":
                    return "NO"
                else:
                    return answer
        return "YES"  # 默认答案
    
    # 第一轮：初始推理
    print(f"\n🎯 第一轮：初始推理")
    print(f"{'='*40}")
    
    prompt_text = f"""Look at the image and answer quickly.

Question: {real_question}

Answer: [Yes or No]
Reason: [Brief reasoning]"""
    
    prompt = [dict(type='text', value=prompt_text)]
    prompt.extend([dict(type='image', value=image_path)])
    response_1 = model.generate(message=prompt)
    answer_1 = extract_answer(response_1)
    
    print(f"  💭 初始推理完成")
    print(f"  📝 初始答案: {answer_1}")
    
    # 记录第一轮
    refine_record['rounds'].append({
        'round': 1,
        'response': response_1,
        'answer': answer_1
    })
    
    # 第二轮：快速自我反思
    print(f"\n🎯 第二轮：快速反思")
    print(f"{'='*40}")
    
    prompt_text = f"""Question: {real_question}

My previous answer: {answer_1}

Quick reflection - should I change my answer?

Final Answer: [Yes or No]
Brief reason: [Quick explanation]"""
    
    prompt = [dict(type='text', value=prompt_text)]
    prompt.extend([dict(type='image', value=image_path)])
    response_2 = model.generate(message=prompt)
    answer_2 = extract_answer(response_2)
    
    print(f"  💭 快速反思完成")
    print(f"  📝 最终答案: {answer_2}")
    
    # 记录第二轮
    refine_record['rounds'].append({
        'round': 2,
        'response': response_2,
        'answer': answer_2
    })
    
    # 分析答案变化
    answer_evolution = [answer_1, answer_2]
    answer_changed = answer_1 != answer_2
    
    print(f"\n📊 答案分析...")
    print(f"  📝 答案序列: {' → '.join(answer_evolution)}")
    if answer_changed:
        print(f"  🔄 答案变化: {answer_1} → {answer_2}")
    else:
        print(f"  ✅ 答案稳定: 无变化")
    
    # 记录结果
    refine_record['final_decision'] = {
        'answer_evolution': answer_evolution,
        'answer_changed': answer_changed,
        'final_answer': answer_2,
        'total_rounds': 2
    }
    
    # 最终结果
    final_result = {
        'method': 'Single Model Self-Refine Fast',
        'final_answer': answer_2,
        'simple_answer': answer_2,
        'question': real_question,
        'ground_truth': base_answer,
        'answer_evolution': answer_evolution,
        'answer_changed': answer_changed,
        'total_rounds': 2,
        'refine_concluded': True
    }
    
    print(f"\n{'🤔'*40}")
    print(f"🤔 单模型Self-Refine快速版本结束")
    print(f"📝 答案演化: {' → '.join(answer_evolution)}")
    print(f"💬 最终答案: {answer_2}")
    print(f"💬 正确答案: {base_answer}")
    if answer_changed:
        print(f"🔄 答案变化: {answer_1} → {answer_2}")
    else:
        print(f"✅ 答案稳定: 无变化")
    print(f"{'🤔'*40}")
    
    # 保存记录
    if save_file:
        refine_save_file = save_file.replace('.json', '_self_refine_fast.json')
        with open(refine_save_file, 'w', encoding='utf-8') as f:
            json.dump(refine_record, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Self-Refine快速版本记录已保存到: {refine_save_file}")
    
    return refine_record, final_result

def baseline_self_refine_option(model, real_question, image_path, base_answer, options=["A", "B", "C", "D", "E"], save_file="", benchmark="MMMU"):
    """
    单模型Self-Refine消融实验 - 支持选项版本
    
    流程：
    1. 第一轮：模型给出初始答案和推理
    2. 第二轮：模型基于第一轮结果进行自我反思和优化
    3. 第三轮：模型给出最终答案
    
    Args:
        model: 使用的语言模型
        real_question: 原始问题
        image_path: 图像路径
        base_answer: 正确答案
        options: 选项列表，如 ["A", "B", "C", "D"] 或 ["Yes", "No"]
        save_file: 保存文件路径
        benchmark: 数据集名称
    
    Returns:
        refine_record: 自我优化记录
        final_result: 最终结果
    """
    print(f"\n🤔 单模型Self-Refine选项版本启动")
    print(f"📝 问题: {real_question}")
    print(f"🎯 正确答案: {base_answer}")
    if options:
        print(f"📋 选项: {options}")
    
    import json
    import re
    
    # 记录
    refine_record = {
        'method': 'Single Model Self-Refine with Options',
        'question': real_question,
        'ground_truth': base_answer,
        'options': options,
        'rounds': []
    }
    
    def extract_answer_with_options(response, options=None):
        """从回复中提取答案选项"""
        if not options:
            # 默认Yes/No模式
            patterns = [
                r'Answer:\s*(Yes|No)',
                r'答案[：:]\s*(是|否|Yes|No)',
                r'选择[：:]\s*(是|否|Yes|No)',
                r'\b(Yes|No)\b',
                r'\b(是|否)\b'
            ]
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    answer = match.group(1).upper()
                    if answer == "是":
                        return "YES"
                    elif answer == "否":
                        return "NO"
                    else:
                        return answer
            return "YES"  # 默认答案
        else:
            if "A" in response:
                return "A"
            elif "B" in response:
                return "B"
            elif "C" in response:
                return "C"
            elif "D" in response:
                return "D"
            elif "E" in response:
                return "E"
            else:
                return response
    
    # 构建选项文本
    if options:
        options_text = "\n".join([f"{opt}. " for opt in options])
        answer_format = f"[{'/'.join(options)}]"
    else:
        options_text = ""
        answer_format = "[Yes or No]"
    
    # 第一轮：初始推理
    print(f"\n🎯 第一轮：初始推理")
    print(f"{'='*40}")
    
    prompt_text = f"""Look at the image and answer the question step by step.

{real_question}

Answer: {answer_format}
Reasoning: [Your step-by-step reasoning]"""
    
    prompt = [dict(type='text', value=prompt_text)]
    prompt.extend([dict(type='image', value=image_path)])
    response_1 = model.generate(message=prompt)
    answer_1 = extract_answer_with_options(response_1, options)
    
    print(f"  💭 初始推理完成")
    print(f"  📝 初始答案: {answer_1}")
    
    # 记录第一轮
    refine_record['rounds'].append({
        'round': 1,
        'response': response_1,
        'answer': answer_1
    })
    
    # 第二轮：自我反思
    print(f"\n🎯 第二轮：自我反思")
    print(f"{'='*40}")
    
    prompt_text = f"""{real_question}

My previous reasoning and answer:
{response_1}

Quick reflection - should I change my answer?

Please provide your reflection and any corrections:
Reflection: [Your self-reflection]
Corrected Answer: {answer_format}
Corrected Reasoning: [Updated reasoning if needed]"""
    
    prompt = [dict(type='text', value=prompt_text)]
    prompt.extend([dict(type='image', value=image_path)])
    response_2 = model.generate(message=prompt)
    answer_2 = extract_answer_with_options(response_2, options)
    
    print(f"  💭 自我反思完成")
    print(f"  📝 反思后答案: {answer_2}")
    
    # 记录第二轮
    refine_record['rounds'].append({
        'round': 2,
        'response': response_2,
        'answer': answer_2
    })
    
    # 第三轮：最终确认
    print(f"\n🎯 第三轮：最终确认")
    print(f"{'='*40}")
    
    prompt_text = f"""{real_question}

My initial reasoning: {response_1}
My reflection: {response_2}

Based on all my analysis, what is my final answer?

Final Answer: {answer_format}
Final Reasoning: [Your conclusive reasoning]"""
    
    prompt = [dict(type='text', value=prompt_text)]
    prompt.extend([dict(type='image', value=image_path)])
    response_3 = model.generate(message=prompt)
    answer_3 = extract_answer_with_options(response_3, options)
    
    print(f"  💭 最终确认完成")
    print(f"  📝 最终答案: {answer_3}")
    
    # 记录第三轮
    refine_record['rounds'].append({
        'round': 3,
        'response': response_3,
        'answer': answer_3
    })
    
    # 分析答案变化
    answer_evolution = [answer_1, answer_2, answer_3]
    answer_changes = []
    for i in range(1, len(answer_evolution)):
        if answer_evolution[i] != answer_evolution[i-1]:
            answer_changes.append(f"Round {i}: {answer_evolution[i-1]} → {answer_evolution[i]}")
    
    print(f"\n📊 答案演化分析...")
    print(f"  📝 答案序列: {' → '.join(answer_evolution)}")
    if answer_changes:
        print(f"  🔄 答案变化: {'; '.join(answer_changes)}")
    else:
        print(f"  ✅ 答案稳定: 无变化")
    
    # 记录结果
    refine_record['final_decision'] = {
        'answer_evolution': answer_evolution,
        'answer_changes': answer_changes,
        'final_answer': answer_3,
        'total_rounds': 3
    }
    
    # 最终结果
    final_result = {
        'method': 'Single Model Self-Refine with Options',
        'final_answer': response_3,
        'simple_answer': answer_3,
        'question': real_question,
        'ground_truth': base_answer,
        'options': options,
        'answer_evolution': answer_evolution,
        'answer_changes': answer_changes,
        'total_rounds': 3,
        'refine_concluded': True
    }
    
    print(f"\n{'🤔'*40}")
    print(f"🤔 单模型Self-Refine选项版本结束")
    print(f"📝 答案演化: {' → '.join(answer_evolution)}")
    print(f"💬 最终答案: {answer_3}")
    print(f"💬 正确答案: {base_answer}")
    if answer_changes:
        print(f"🔄 答案变化: {'; '.join(answer_changes)}")
    else:
        print(f"✅ 答案稳定: 无变化")
    print(f"{'🤔'*40}")
    
    # 保存记录
    if save_file:
        refine_save_file = save_file.replace('.json', '_self_refine_options.json')
        with open(refine_save_file, 'w', encoding='utf-8') as f:
            json.dump(refine_record, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Self-Refine选项版本记录已保存到: {refine_save_file}")
    
    return refine_record, final_result


def baseline_mad_debate_maj(model, real_question, image_path, base_answer, save_file="", benchmark="MMMU"):
    """
    三个agent辩论2轮 + 多数决定 (Yes/No问题)
    
    流程：
    1. 第一轮：3个agent独立分析
    2. 第二轮：3个agent看到其他agent第一轮答案后给出最终答案
    3. 多数决：从第二轮答案中提取Yes/No选项，选择最多的作为最终答案
    
    Args:
        model: 使用的语言模型
        real_question: 原始问题 (Yes/No问题)
        image_path: 图像路径
        base_answer: 正确答案 (Yes/No)
        save_file: 保存文件路径
        benchmark: 数据集名称
    
    Returns:
        debate_record: 辩论记录
        final_result: 最终结果
    """
    print(f"\n⚔️ 三Agent Yes/No辩论系统启动")
    print(f"📝 问题: {real_question}")
    print(f"🎯 正确答案: {base_answer}")
    
    import json
    import re
    from collections import Counter
    
    # 记录
    debate_record = {
        'method': '3-Agent Yes/No Debate',
        'question': real_question,
        'ground_truth': base_answer,
        'rounds': []
    }
    
    # 存储每轮答案
    agent_responses = {
        'agent_1': [],
        'agent_2': [], 
        'agent_3': []
    }
    
    def extract_answer(response):
        """从回复中提取答案选项"""
        patterns = [
            r'Answer:\s*(Yes|No)',
            r'答案[：:]\s*(是|否|Yes|No)',
            r'选择[：:]\s*(是|否|Yes|No)',
            r'\b(Yes|No)\b',
            r'\b(是|否)\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                # 处理中文答案
                if answer == "是":
                    return "YES"
                elif answer == "否":
                    return "NO"
                else:
                    return answer
        return "YES"  # 默认答案
    
    # 第一轮：独立分析
    print(f"\n🎯 第一轮：独立分析")
    print(f"{'='*40}")
    
    for i in range(1, 4):
        print(f"  💭 Agent {i} 分析中...")
        prompt_text = f"""Analyze this image and answer the question.

Question: {real_question}

Format:
Answer: [Yes or No]
Reasoning: [Brief explanation]"""
        
#         prompt = [dict(type='text', value=prompt_text)]
#         prompt.extend([dict(type='image', value=image_path)])
#         response = model.generate(message=prompt)
        response = model.generate([image_path, prompt_text])
        agent_responses[f'agent_{i}'].append(response)
        print(f"  ✅ Agent {i} 完成")
    
    # 记录第一轮
    debate_record['rounds'].append({
        'round': 1,
        'agent_1': agent_responses['agent_1'][0],
        'agent_2': agent_responses['agent_2'][0],
        'agent_3': agent_responses['agent_3'][0]
    })
    
    # 第二轮：最终答辩轮（参考其他意见后给出最终答案）
    print(f"\n🎯 第二轮：最终答辩")
    print(f"{'='*40}")
    
    for i in range(1, 4):
        print(f"  💭 Agent {i} 最终答辩中...")
        
        # 获取其他两个agent的第一轮答案
        other_agents = [j for j in range(1, 4) if j != i]
        other_responses = "\n".join([f"Agent {j}: {agent_responses[f'agent_{j}'][0]}" for j in other_agents])
        
        prompt_text = f"""Question: {real_question}

Other analysts' opinions from round 1:
{other_responses}

This is the final round. Consider the above opinions and provide your final answer:
Answer: [Yes or No]
Reasoning: [Brief explanation]"""
        
#         prompt = [dict(type='text', value=prompt_text)]
#         prompt.extend([dict(type='image', value=image_path)])
#         response = model.generate(message=prompt)
        response = model.generate([image_path, prompt_text])
        agent_responses[f'agent_{i}'].append(response)
        print(f"  ✅ Agent {i} 答辩完成")
    
    # 记录第二轮
    debate_record['rounds'].append({
        'round': 2,
        'agent_1': agent_responses['agent_1'][1],
        'agent_2': agent_responses['agent_2'][1],
        'agent_3': agent_responses['agent_3'][1]
    })
    
    # 统计第二轮答案结果
    print(f"\n📊 统计最终答案...")
    final_answers = []
    for i in range(1, 4):
        answer = extract_answer(agent_responses[f'agent_{i}'][1])
        final_answers.append(answer)
        print(f"  📝 Agent {i} 答案: {answer}")
    
    # 计算最终答案（多数决）
    answer_counter = Counter(final_answers)
    most_common = answer_counter.most_common(1)[0]
    final_answer = most_common[0]
    answer_count = most_common[1]
    
    print(f"  📈 答案统计: {dict(answer_counter)}")
    print(f"  🏆 最终答案: {final_answer} (出现: {answer_count}/3次)")
    
    # 记录结果
    debate_record['final_decision'] = {
        'answers': final_answers,
        'answer_distribution': dict(answer_counter),
        'final_answer': final_answer,
        'answer_count': answer_count
    }
    
    # 最终结果
    final_result = {
        'method': '3-Agent Yes/No Debate',
        'final_answer': final_answer,
        'simple_answer': final_answer,
        'question': real_question,
        'ground_truth': base_answer,
        'answers': final_answers,
        'answer_distribution': dict(answer_counter),
        'total_rounds': 2,
        'debate_concluded': True
    }
    
    print(f"\n{'🏆'*40}")
    print(f"⚖️ 三Agent Yes/No辩论结束")
    print(f"📝 各Agent答案: {' '.join(final_answers)}")
    print(f"💬 最终答案: {final_answer}")
    print(f"💬 正确答案: {base_answer}")
    print(f"📊 答案统计: {dict(answer_counter)}")
    print(f"{'🏆'*40}")
    
    # 保存记录
    if save_file:
        debate_save_file = save_file.replace('.json', '_3agent_yesno_debate.json')
        with open(debate_save_file, 'w', encoding='utf-8') as f:
            json.dump(debate_record, f, ensure_ascii=False, indent=2)
        print(f"\n💾 三Agent Yes/No辩论记录已保存到: {debate_save_file}")
    
    return debate_record, final_result


def baseline_mad_debate(model, real_question, image_path, base_answer, save_file="", benchmark="MMMU"):
    """
    三个agent辩论3轮 + 法官决策 (Yes/No问题) - 简化版本
    
    流程：
    1. 第一轮：3个agent快速分析
    2. 第二轮：3个agent简单参考其他意见
    3. 第三轮：3个agent给出最终答案
    4. 法官决策：法官简单选择一个答案
    
    Args:
        model: 使用的语言模型
        real_question: 原始问题 (Yes/No问题)
        image_path: 图像路径
        base_answer: 正确答案 (Yes/No)
        save_file: 保存文件路径
        benchmark: 数据集名称
    
    Returns:
        debate_record: 辩论记录
        final_result: 最终结果
    """
    print(f"\n⚔️ 三Agent+法官 Yes/No辩论系统启动")
    print(f"📝 问题: {real_question}")
    print(f"🎯 正确答案: {base_answer}")
    
    import json
    import re
    from collections import Counter
    
    # 记录
    debate_record = {
        'method': '3-Agent Yes/No Debate with Judge',
        'question': real_question,
        'ground_truth': base_answer,
        'rounds': []
    }
    
    # 存储每轮答案
    agent_responses = {
        'agent_1': [],
        'agent_2': [], 
        'agent_3': []
    }
    
    def extract_answer(response):
        """从回复中提取答案选项"""
        patterns = [
            r'Answer:\s*(Yes|No)',
            r'答案[：:]\s*(是|否|Yes|No)',
            r'选择[：:]\s*(是|否|Yes|No)',
            r'\b(Yes|No)\b',
            r'\b(是|否)\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                # 处理中文答案
                if answer == "是":
                    return "YES"
                elif answer == "否":
                    return "NO"
                else:
                    return answer
        return "YES"  # 默认答案
    
    # 第一轮：快速分析
    print(f"\n🎯 第一轮：快速分析")
    print(f"{'='*40}")
    
    for i in range(1, 4):
        print(f"  💭 Agent {i} 分析中...")
        prompt_text = f"""Look at this image and answer quickly.

Question: {real_question}

Answer: [Yes or No]
Reason: [One sentence only]"""
        
        prompt = [dict(type='text', value=prompt_text)]
        prompt.extend([dict(type='image', value=image_path)])
        response = model.generate(message=prompt)
        agent_responses[f'agent_{i}'].append(response)
        print(f"  ✅ Agent {i} 完成")
    
    # 记录第一轮
    debate_record['rounds'].append({
        'round': 1,
        'agent_1': agent_responses['agent_1'][0],
        'agent_2': agent_responses['agent_2'][0],
        'agent_3': agent_responses['agent_3'][0]
    })
    
    # 第二轮：简单参考
    print(f"\n🔥 第二轮：简单参考")
    print(f"{'='*40}")
    
    for i in range(1, 4):
        print(f"  💭 Agent {i} 分析中...")
        
        # 获取其他两个agent的答案（仅答案，不包括推理）
        other_agents = [j for j in range(1, 4) if j != i]
        other_answers = [extract_answer(agent_responses[f'agent_{j}'][0]) for j in other_agents]
        other_summary = f"Others said: {', '.join(other_answers)}"
        
        prompt_text = f"""Question: {real_question}

{other_summary}

Your answer:
Answer: [Yes or No]
Reason: [Brief]"""
        
        prompt = [dict(type='text', value=prompt_text)]
        prompt.extend([dict(type='image', value=image_path)])
        response = model.generate(message=prompt)
        agent_responses[f'agent_{i}'].append(response)
        print(f"  ✅ Agent {i} 完成")
    
    # 记录第二轮
    debate_record['rounds'].append({
        'round': 2,
        'agent_1': agent_responses['agent_1'][1],
        'agent_2': agent_responses['agent_2'][1],
        'agent_3': agent_responses['agent_3'][1]
    })
    
    # 第三轮：最终答案
    print(f"\n🎯 第三轮：最终答案")
    print(f"{'='*40}")
    
    for i in range(1, 4):
        print(f"  💭 Agent {i} 最终答辩中...")
        
        prompt_text = f"""Question: {real_question}

Give your final answer:
Answer: [Yes or No]
Reason: [Short]"""
        
        prompt = [dict(type='text', value=prompt_text)]
        prompt.extend([dict(type='image', value=image_path)])
        response = model.generate(message=prompt)
        agent_responses[f'agent_{i}'].append(response)
        print(f"  ✅ Agent {i} 答辩完成")
    
    # 记录第三轮
    debate_record['rounds'].append({
        'round': 3,
        'agent_1': agent_responses['agent_1'][2],
        'agent_2': agent_responses['agent_2'][2],
        'agent_3': agent_responses['agent_3'][2]
    })
    
    # 收集第三轮答案
    print(f"\n📊 收集各Agent最终答案...")
    final_answers = []
    agent_final_responses = []
    for i in range(1, 4):
        answer = extract_answer(agent_responses[f'agent_{i}'][2])
        final_answers.append(answer)
        agent_final_responses.append(agent_responses[f'agent_{i}'][2])
        print(f"  📝 Agent {i} 答案: {answer}")
    
    # 法官快速决策
    print(f"\n⚖️ 法官快速决策")
    print(f"{'='*40}")
    print(f"  👨‍⚖️ 法官快速选择...")
    
    # 简化的法官prompt
    agent_answers_only = [f"Agent {i+1}: {final_answers[i]}" for i in range(3)]
    
    judge_prompt_text = f"""Question: {real_question}

Three answers: {' | '.join(agent_answers_only)}

Pick one answer quickly:
Selected Answer: [Yes or No]
Chosen Agent: [Agent 1, Agent 2, or Agent 3]
Why: [One sentence]"""
    
    judge_prompt = [dict(type='text', value=judge_prompt_text)]
    judge_prompt.extend([dict(type='image', value=image_path)])
    judge_response = model.generate(message=judge_prompt)
    
    # 提取法官的决定
    final_answer = extract_answer(judge_response)
    
    # 确定被选中的agent
    chosen_agent_match = re.search(r'Chosen Agent:\s*Agent\s*([123])', judge_response, re.IGNORECASE)
    chosen_agent = chosen_agent_match.group(1) if chosen_agent_match else "1"
    
    print(f"  ✅ 法官评审完成")
    print(f"  🏆 法官选择: Agent {chosen_agent}")
    print(f"  📋 最终答案: {final_answer}")
    
    # 统计信息（保留用于分析）
    answer_counter = Counter(final_answers)
    print(f"  📈 各Agent答案统计: {dict(answer_counter)}")
    
    # 记录法官轮次
    debate_record['judge_round'] = {
        'judge_response': judge_response,
        'chosen_agent': chosen_agent,
        'final_decision': final_answer
    }
    
    # 记录结果
    debate_record['final_decision'] = {
        'agent_answers': final_answers,
        'answer_distribution': dict(answer_counter),
        'judge_response': judge_response,
        'chosen_agent': chosen_agent,
        'final_answer': final_answer,
        'decision_method': 'judge_selection'
    }
    
    # 最终结果
    final_result = {
        'method': '3-Agent Yes/No Debate with Judge',
        'final_answer': final_answer,
        'simple_answer': final_answer,
        'question': real_question,
        'ground_truth': base_answer,
        'agent_answers': final_answers,
        'answer_distribution': dict(answer_counter),
        'chosen_agent': chosen_agent,
        'judge_response': judge_response,
        'total_rounds': 3,
        'decision_method': 'judge_selection',
        'debate_concluded': True
    }
    
    print(f"\n{'🏆'*40}")
    print(f"⚖️ 三Agent Yes/No辩论+法官决策结束")
    print(f"📝 各Agent答案: {' '.join(final_answers)}")
    print(f"👨‍⚖️ 法官选择: Agent {chosen_agent}")
    print(f"💬 最终答案: {final_answer}")
    print(f"💬 正确答案: {base_answer}")
    print(f"📊 答案统计: {dict(answer_counter)}")
    print(f"{'🏆'*40}")
    
    # 保存记录
#     if save_file:
#         debate_save_file = save_file.replace('.json', '_3agent_judge_debate.json')
#         with open(debate_save_file, 'w', encoding='utf-8') as f:
#             json.dump(debate_record, f, ensure_ascii=False, indent=2)
#         print(f"\n💾 三Agent+法官决策记录已保存到: {debate_save_file}")
    
    return debate_record, final_result