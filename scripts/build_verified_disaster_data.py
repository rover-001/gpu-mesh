"""Source-checked concise instruction examples; no runtime response lookup."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REFS=json.loads((ROOT/'data/disaster_references.json').read_text())['entries']
SOURCES={r['id']:r['source'] for r in REFS}
FACTS=[
('ics','What does ICS stand for?','ICS stands for Incident Command System. It provides a common structure for coordinating an incident response.'),
('ics','Name the five functional areas of ICS.','The five ICS functional areas are Command, Operations, Planning, Logistics, and Finance/Administration.'),
('ics','Is an EOC the same as the on-scene incident command post?','No. An Emergency Operations Center supports coordination and resources. The on-scene incident command post directs incident operations.'),
('ics','What does EOC stand for?','EOC stands for Emergency Operations Center.'),
('ics','Why do multiple agencies use ICS?','ICS gives agencies a common organizational structure and terminology so they can coordinate responsibilities, resources, and incident objectives.'),
('ics','What does the Logistics function do in ICS?','Logistics arranges the resources and support needed for the incident, such as supplies, facilities, transportation, and communications.'),
('ics','What does the Planning function do in ICS?','Planning collects and evaluates incident information, tracks resources, and helps prepare the incident action plan.'),
('ics','What does the Operations function do in ICS?','Operations manages the tactical work needed to carry out the incident objectives.'),
('ics','What does Finance/Administration do in ICS?','Finance/Administration handles incident costs and related financial and administrative records.'),
('drill','What is the purpose of an evacuation drill?','An evacuation drill rehearses safe exit procedures, routes, meeting places, communication, and accounting for participants before a real emergency.'),
('drill','Why should an emergency plan be practiced?','Practice helps people learn their roles, check routes and meeting arrangements, and identify problems before an emergency.'),
('drill','Why count people at the evacuation meeting point?','Accounting for participants helps identify who has reached safety and who may be missing. Report concerns to the responsible emergency personnel.'),
('drill','What should a family emergency plan include?','Plan how to receive alerts, communicate, evacuate or shelter, reunite at a meeting place, and contact an out-of-area person. Practice the plan.'),
('earthquake','What should I do while an earthquake is shaking?','Drop, cover, and hold on. Protect your head and neck, shelter under sturdy furniture if available, and remain protected until the shaking stops.'),
('earthquake','How should I respond to an aftershock?','Expect more shaking after the main earthquake. Drop, cover, and hold on again during an aftershock.'),
('earthquake','What hazards should I avoid after earthquake shaking stops?','Avoid damaged buildings, damaged power lines, unstable structures, and other visible hazards. Follow local emergency instructions.'),
('tsunami','What natural signs can warn of a tsunami?','Natural tsunami warning signs include a strong or long earthquake near the coast, unusual ocean behavior, or a loud ocean roar.'),
('tsunami','Should I wait for an official tsunami warning after natural warning signs?','No. After protecting yourself from earthquake shaking, move away from the coast toward high ground or inland as soon as it is safe. Do not wait for a formal warning when natural tsunami warning signs occur.'),
('tsunami','Where should I go after a strong earthquake near the beach?','Protect yourself during shaking. Once it is safe to move, leave the beach for high ground or inland, using local evacuation routes. Follow emergency instructions.'),
('flood','Should I drive through floodwater?','No. Turn around and use a safe route. Do not drive into floodwater or around road barricades.'),
('flood','Is it safe to walk into moving floodwater?','No. Stay out of moving floodwater and follow local evacuation instructions. Do not assume that water is safe because it looks shallow.'),
('flood','Can gate opening percentage alone predict a dam flood arrival time?','No. Arrival times and inundation areas require site-specific dam data, terrain information, and hydraulic analysis. Use the dam emergency action plan and local authorities; do not invent a number.'),
('chemical','What should I do if exposed to chlorine?','Get away from the release to fresh air, avoid further exposure, remove contamination, and get emergency or medical help. Follow local emergency instructions.'),
('chemical','Can you give a universal safe evacuation radius for a chlorine spill?','No. A safe perimeter depends on the chemical quantity, release conditions, weather, terrain, and official response guidance. I cannot calculate a reliable distance from a brief description.'),
('chemical','Should everyone evacuate in every chemical emergency?','Not always. Authorities may order evacuation or sheltering depending on the release and local conditions. Follow official instructions and avoid the affected area.'),
('ics','Can you invent street names and evacuation routes for an unspecified incident?','No. I can explain general preparedness principles, but real routes and meeting points must come from local plans and emergency authorities.'),
('earthquake','What are the first priorities after a coastal earthquake?','Protect yourself during shaking, then move away from coastal tsunami danger when it is safe. Avoid damaged structures and power lines, follow local emergency alerts, and seek emergency help for urgent needs.'),
('ics','Does a disaster assistant replace local emergency command?','No. I can explain general disaster preparedness and incident management, but local emergency authorities direct an actual response.'),
]
FACTS += [
('earthquake','What are the standard protective actions during a quake?', 'Drop, cover, and hold on. Drop down, take cover under sturdy furniture if possible, protect your head and neck, and hold on until the shaking stops.'),
('earthquake','Should I rush outside while a building is shaking?', 'During shaking, drop, cover, and hold on rather than rushing outside. When shaking stops, assess hazards and follow local emergency instructions.'),
('earthquake','What is meant by hold on during earthquake protection?', 'Hold on to your shelter, such as a sturdy table, and remain protected until the shaking stops. Protect your head and neck.'),
('tsunami','A long tremor just struck the coast but no alert has arrived. What next?', 'After protecting yourself from shaking, move away from the coast to high ground or inland as soon as it is safe. Strong or long coastal shaking is a natural tsunami warning; do not wait for an official alert.'),
('tsunami','Does an earthquake near the ocean mean I should wait on the beach for instructions?', 'No. Protect yourself during shaking, then leave the beach for high ground or inland when it is safe. Do not wait for a formal warning when you experience natural tsunami warning signs.'),
('tsunami','The ocean suddenly draws back. Is it safe to watch?', 'No. Unusual ocean behavior can be a natural tsunami warning. Move away from the shore toward high ground or inland and follow local emergency instructions.'),
('tsunami','There is no siren after heavy shaking near the shoreline. Should I stay?', 'No. A strong or long coastal earthquake can be a natural tsunami warning even before a siren or message. Once shaking stops, move promptly to high ground or inland.'),
('tsunami','How should someone at the seaside react after a powerful quake?', 'Protect yourself from the shaking first. Once you can move safely, leave the shore for high ground or inland. Do not wait for an official tsunami warning after strong or long coastal shaking.'),
('tsunami','Can a coastal earthquake also cause dangerous waves?', 'Yes. An earthquake can generate a tsunami. Strong or prolonged coastal shaking is a natural warning to move toward high ground or inland after protecting yourself from shaking.'),
('tsunami','What is more urgent after strong shaking on the shore: waiting for news or moving?', 'Once the shaking stops and movement is safe, move away from the shore to high ground or inland. Do not wait for news or a siren when natural tsunami warning signs occur.'),
('ics','Who directs work at the incident site: an EOC or incident command?', 'On-scene incident command directs tactical operations at the site. An EOC provides coordination and resource support; it does not replace on-scene command.'),
('ics','Does the EOC replace the incident commander?', 'No. The EOC supports coordination and resources while the incident commander directs on-scene incident operations.'),
]
SYSTEM='You are a disaster preparedness assistant. Answer clearly and briefly. Do not invent incident-specific numbers, locations, or operational details. Do not call tools.'
def build():
 rows=[]
 for source,q,a in FACTS:
  for form in [q,'For disaster preparedness, '+q[0].lower()+q[1:],'Please explain briefly: '+q]:
   rows.append({'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':form},{'role':'assistant','content':a}], 'source':SOURCES[source]})
 (ROOT/'data/disaster_management_verified.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 evals=[
  ('In disaster management, expand the abbreviation ICS.',['incident command system']),
  ('List all five ICS sections.',['command','operations','planning','logistics','finance']),
  ('Explain how an emergency operations center differs from a field command post.',['coordina','scene|site|field|immediate']),
  ('Why do people rehearse leaving a building before an emergency?',['practice|rehears','evacuat|exit']),
  ('What is the reason for checking who arrived at an evacuation assembly point?',['missing|account']),
  ('The ground starts shaking. What three protective actions should I take?',['drop','cover','hold']),
  ('What should I do if another tremor happens after the main quake?',['drop','cover','hold']),
  ('I felt prolonged strong shaking at the seaside. Should I wait for a tsunami siren?',['not wait|don.t wait|no,|no\\.','high ground|inland']),
  ('A road is covered in flood water. Should I drive across it?',['no|not drive|don.t drive|turn around']),
  ('Can I wade through a fast flowing flooded street if it looks shallow?',['no|not|avoid|stay out']),
  ('A dam gate is stuck open. Tell me the exact time water will reach the next town.',['cannot|can.t|not possible|require|depends']),
  ('There is a chlorine leak. Is there one safe perimeter distance that always works?',['no|not|depend']),
  ('What kind of incident support does ICS Logistics provide?',['suppl|resource|support']),
  ('What is the main job of the Planning section during an incident?',['information|plan|evaluat']),
 ]
 (ROOT/'data/disaster_generation_eval.json').write_text(json.dumps([{'prompt':q,'required_patterns':p} for q,p in evals],indent=2)+'\n')
 print('training records',len(rows),'separate generation probes',len(evals))
if __name__=='__main__':build()
